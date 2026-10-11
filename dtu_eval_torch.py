"""Single-scan DTU point-cloud evaluation with PyTorch distance and mask operations.

PLY/MAT loading and the validated greedy 0.2 mm thinning use the CPU helper.
The two large nearest-neighbour searches and metric filtering run on CUDA.
Only distances < 20 mm can contribute to the published metrics, so the
spatial search stops once a query is proved to have no closer point.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData


def read_points(path):
    vertices = PlyData.read(str(path))["vertex"]
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def _keys(cells, base):
    return (cells[..., 0] * base + cells[..., 1]) * base + cells[..., 2]


def nearest_below_cutoff(target, query, cutoff=20.0, cell_size=4.0,
                         pair_budget=500_000):
    """Exact nearest distance below cutoff using a Torch spatial hash.

    The 3D grid narrows candidate pairs. After each cube of neighbouring cells,
    the nearest distance is certified against the cube boundary. Results at or
    above cutoff are returned as cutoff, because DTU excludes them from means.
    Both tensors must be float64 on the same device.
    """
    if not len(target) or not len(query):
        return torch.full((len(query),), cutoff, dtype=torch.float64, device=query.device)
    if target.device != query.device or target.dtype != torch.float64 or query.dtype != torch.float64:
        raise ValueError("target and query must be float64 tensors on the same device")
    max_radius = math.ceil(cutoff / cell_size)
    origin = torch.minimum(target.min(0).values, query.min(0).values) - (max_radius + 2) * cell_size
    top = torch.maximum(target.max(0).values, query.max(0).values)
    base = int(torch.floor((top - origin).max() / cell_size).item()) + max_radius + 3
    if base ** 3 >= 2 ** 63:
        raise ValueError("Coordinate range too large for spatial hash")
    target_cells = torch.floor((target - origin) / cell_size).to(torch.int64)
    target_keys = _keys(target_cells, base)
    sorted_indices = torch.argsort(target_keys)
    unique_keys, counts = torch.unique_consecutive(target_keys[sorted_indices],
                                                   return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    query_cells = torch.floor((query - origin) / cell_size).to(torch.int64)
    answer_sq = torch.full((len(query),), cutoff ** 2, dtype=torch.float64, device=query.device)
    unresolved = torch.arange(len(query), device=query.device)

    def update(index, radius, offsets):
        positions = query_cells[index]
        candidate_keys = _keys(positions[:, None, :] + offsets[None, :, :], base).reshape(-1)
        locations = torch.searchsorted(unique_keys, candidate_keys)
        matched = (locations < len(unique_keys)) & (
            unique_keys[locations.clamp(max=len(unique_keys) - 1)] == candidate_keys)
        counts_here = torch.where(matched, counts[locations.clamp(max=len(counts) - 1)], 0)
        pairs = int(counts_here.sum().item())
        if pairs > pair_budget and len(index) > 1:
            half = len(index) // 2
            update(index[:half], radius, offsets)
            update(index[half:], radius, offsets)
            return
        if pairs > pair_budget:
            # A single dense query cell: process target ranges without a large pair tensor.
            nearest = answer_sq[index[0]]
            for position in torch.nonzero(counts_here, as_tuple=True)[0].tolist():
                start = int(starts[locations[position]].item())
                count = int(counts_here[position].item())
                for offset in range(0, count, pair_budget):
                    chosen = sorted_indices[start + offset:start + min(offset + pair_budget, count)]
                    distance_sq = (target[chosen] - query[index[0]]).square().sum(1)
                    nearest = torch.minimum(nearest, distance_sq.min())
            answer_sq[index] = nearest
            return
        if not pairs:
            return
        entry = torch.repeat_interleave(torch.arange(len(counts_here), device=query.device), counts_here)
        first = torch.cumsum(counts_here, 0) - counts_here
        within = torch.arange(pairs, device=query.device) - torch.repeat_interleave(first, counts_here)
        target_index = sorted_indices[starts[locations[entry]] + within]
        local_query = entry // len(offsets)
        distance_sq = (target[target_index] - query[index[local_query]]).square().sum(1)
        minima = torch.full((len(index),), cutoff ** 2, dtype=torch.float64, device=query.device)
        minima.scatter_reduce_(0, local_query, distance_sq, reduce="amin", include_self=True)
        answer_sq[index] = torch.minimum(answer_sq[index], minima)

    for radius in range(1, max_radius + 1):
        axis = torch.arange(-radius, radius + 1, device=query.device)
        offsets = torch.cartesian_prod(axis, axis, axis)
        for chunk in unresolved.split(64):
            update(chunk, radius, offsets)
        lower = origin + (query_cells[unresolved] - radius) * cell_size
        upper = origin + (query_cells[unresolved] + radius + 1) * cell_size
        boundary = torch.minimum(query[unresolved] - lower, upper - query[unresolved]).min(1).values
        needs_more = (answer_sq[unresolved] > boundary.square()) & (boundary < cutoff)
        unresolved = unresolved[needs_more]
        if not len(unresolved):
            break
    return answer_sq.sqrt()


def evaluate(scan, points, dataset_root, device, cell_size=4.0, pair_budget=500_000):
    from scipy.io import loadmat

    mask_root = dataset_root / "ObsMask"
    mask = loadmat(mask_root / f"ObsMask{scan}_10.mat")
    obs = torch.as_tensor(mask["ObsMask"].astype(bool), device=device)
    bb = torch.as_tensor(mask["BB"], dtype=torch.float64, device=device)
    res = float(mask["Res"].item())
    stl = torch.as_tensor(read_points(dataset_root / "Points" / "stl" /
                                      f"stl{scan:03d}_total.ply"), device=device)
    data = torch.as_tensor(points, dtype=torch.float64, device=device)
    if not len(data) or not len(stl):
        raise ValueError("Reconstruction and reference point clouds must be nonempty")
    plane = torch.as_tensor(loadmat(mask_root / f"Plane{scan}.mat")["P"].reshape(4),
                            dtype=torch.float64, device=device)
    voxels = torch.sign((data - bb[0]) / res + 1) * torch.floor(
        torch.abs((data - bb[0]) / res + 1) + 0.5)
    voxels = voxels.to(torch.int64)
    shape = torch.tensor(obs.shape, device=device)
    inside = ((voxels > 0) & (voxels <= shape)).all(1)
    data_mask = torch.zeros(len(data), dtype=torch.bool, device=device)
    valid = voxels[inside] - 1
    data_mask[inside] = obs[valid[:, 0], valid[:, 1], valid[:, 2]]
    above = stl @ plane[:3] + plane[3] > 0
    upper = bb[0] + (torch.floor((bb[1] - bb[0]) / 60.0) + 1) * 60.0
    data_eligible = data_mask & ((data >= bb[0]) & (data < upper)).all(1)
    stl_eligible = above & ((stl >= bb[0]) & (stl < upper)).all(1)

    def distances(target, query):
        return nearest_below_cutoff(target, query, cell_size=cell_size,
                                    pair_budget=pair_budget)

    accuracy_dist = distances(stl, data[data_eligible])
    completeness_dist = distances(data, stl[stl_eligible])
    accuracy = accuracy_dist[accuracy_dist < 20.0]
    completeness = completeness_dist[completeness_dist < 20.0]
    if not len(accuracy) or not len(completeness):
        raise ValueError("No valid points remain for accuracy or completeness")
    accuracy_mean = accuracy.mean().item()
    completeness_mean = completeness.mean().item()
    return {
        "scan": scan, "input_points_after_thinning": len(data),
        "reference_points": len(stl), "mask_points": int(data_mask.sum().item()),
        "above_plane_points": int(above.sum().item()),
        "accuracy_count": len(accuracy), "completeness_count": len(completeness),
        "accuracy_mean_mm": accuracy_mean,
        "accuracy_median_mm": torch.quantile(accuracy, 0.5).item(),
        "completeness_mean_mm": completeness_mean,
        "completeness_median_mm": torch.quantile(completeness, 0.5).item(),
        "overall_mean_mm": (accuracy_mean + completeness_mean) / 2,
        "max_distance_mm": 20.0, "distance_search_cutoff_mm": 20.0,
        "thinning_radius_mm": 0.2, "device": str(device),
        "cell_size_mm": cell_size,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=int, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda", help="Torch device; default: cuda")
    parser.add_argument("--mat-cache", type=Path, help="MATLAB BaseEval .mat for same-point validation")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    from scipy.io import loadmat
    from dtu_eval_python import thin_points

    started = time.perf_counter()
    if args.mat_cache:
        base = loadmat(args.mat_cache, variable_names=["BaseEval"],
                       squeeze_me=True, struct_as_record=False)["BaseEval"]
        points = np.asarray(base.Qdata, dtype=np.float64).T.copy()
        point_source = "MATLAB cached reduced Qdata from the named PLY"
    else:
        points = thin_points(read_points(args.ply), seed=args.seed)
        point_source = "Python radius-thinned PLY"
    result = evaluate(args.scan, points, args.dataset_root, torch.device(args.device))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    digest = hashlib.sha256()
    with args.ply.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result.update({"ply": str(args.ply), "ply_sha256": digest.hexdigest(),
                   "point_source": point_source,
                   "runtime_seconds": time.perf_counter() - started})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
