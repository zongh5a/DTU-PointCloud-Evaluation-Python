"""DTU point-cloud evaluation matching the supplied MATLAB PointCompareMain.

Uses the same 0.2 mm stochastic radius thinning, 60 mm nearest-neighbor cap,
ObsMask/ground-plane filtering, and 20 mm reporting threshold. The MATLAB
thinning permutation is not bitwise reproducible in NumPy; --mat-cache uses
MATLAB's stored reduced points for a strict distance/mask comparison.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData
from scipy.io import loadmat
from scipy.spatial import cKDTree


def read_points(path):
    vertices = PlyData.read(str(path))["vertex"]
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def thin_points(points, radius=0.2, seed=0, chunk_size=8192):
    """Greedy random-order radius thinning, as in reducePts_haa.m."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    keep = np.ones(len(points), dtype=bool)
    tree = cKDTree(points)
    for start in range(0, len(points), chunk_size):
        indices = order[start:start + chunk_size]
        neighbors = tree.query_ball_point(points[indices], radius, workers=-1)
        for index, nearby in zip(indices, neighbors):
            if keep[index]:
                keep[nearby] = False
                keep[index] = True
        if (start // chunk_size) % 100 == 0:
            print(json.dumps({"phase": "thin", "processed": min(start + chunk_size, len(points)), "total": len(points)}), flush=True)
    return points[keep]


def matlab_round(values):
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def capped_nearest(target, query, bb, cap=60.0, chunk_size=200000):
    """Equivalent to MaxDistCP for distances < cap; out-of-box points stay cap."""
    dist = np.full(len(query), cap, dtype=np.float64)
    ranges = np.floor((bb[1] - bb[0]) / cap)
    upper = bb[0] + (ranges + 1) * cap
    eligible = np.all((query >= bb[0]) & (query < upper), axis=1)
    indices = np.flatnonzero(eligible)
    tree = cKDTree(target)
    for start in range(0, len(indices), chunk_size):
        chosen = indices[start:start + chunk_size]
        result = tree.query(query[chosen], k=1, distance_upper_bound=cap, workers=-1)[0]
        dist[chosen] = np.minimum(result, cap)
        print(json.dumps({"phase": "nearest", "processed": min(start + chunk_size, len(indices)), "total": len(indices)}), flush=True)
    return dist


def evaluate(scan, points, dataset_root, max_dist=20.0):
    mask_root = dataset_root / "ObsMask"
    mask = loadmat(mask_root / f"ObsMask{scan}_10.mat")
    obs, bb, res = mask["ObsMask"].astype(bool), mask["BB"].astype(np.float64), float(mask["Res"].item())
    stl = read_points(dataset_root / "Points" / "stl" / f"stl{scan:03d}_total.ply")
    plane = loadmat(mask_root / f"Plane{scan}.mat")["P"].reshape(4)

    ddata = capped_nearest(stl, points, bb)
    dstl = capped_nearest(points, stl, bb)

    voxel = matlab_round((points - bb[0]) / res + 1).astype(np.int64)
    inside = np.all((voxel > 0) & (voxel <= np.array(obs.shape)), axis=1)
    data_mask = np.zeros(len(points), dtype=bool)
    valid = voxel[inside] - 1
    data_mask[np.flatnonzero(inside)] = obs[valid[:, 0], valid[:, 1], valid[:, 2]]
    above = stl @ plane[:3] + plane[3] > 0

    accuracy = ddata[data_mask & (ddata < max_dist)]
    completeness = dstl[above & (dstl < max_dist)]
    result = {"scan": scan, "input_points_after_thinning": len(points), "reference_points": len(stl),
              "mask_points": int(data_mask.sum()), "above_plane_points": int(above.sum()),
              "accuracy_count": len(accuracy), "completeness_count": len(completeness),
              "accuracy_mean_mm": float(accuracy.mean()), "accuracy_median_mm": float(np.median(accuracy)),
              "completeness_mean_mm": float(completeness.mean()), "completeness_median_mm": float(np.median(completeness)),
              "overall_mean_mm": float((accuracy.mean() + completeness.mean()) / 2),
              "max_distance_mm": max_dist, "nearest_cap_mm": 60.0, "thinning_radius_mm": 0.2}
    return result, ddata, dstl, data_mask, above


def compare_with_mat(result, arrays, mat_file):
    base = loadmat(mat_file, variable_names=["BaseEval"], squeeze_me=True, struct_as_record=False)["BaseEval"]
    comparison = {}
    for name, actual, reference in [
        ("Ddata", arrays[0], base.Ddata), ("Dstl", arrays[1], base.Dstl),
        ("DataInMask", arrays[2], base.DataInMask),
        ("StlAbovePlane", arrays[3], base.StlAbovePlane),
    ]:
        ref = np.asarray(reference).reshape(-1)
        if actual.dtype == bool:
            comparison[name] = {"mismatched": int(np.count_nonzero(actual != ref.astype(bool))), "total": len(actual)}
        else:
            delta = np.abs(actual - ref)
            comparison[name] = {"mean_abs_diff_mm": float(delta.mean()), "max_abs_diff_mm": float(delta.max()),
                                "over_1e-6_mm": int(np.count_nonzero(delta > 1e-6)), "total": len(actual)}
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, required=True)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mat-cache", type=Path, help="MATLAB BaseEval .mat, for strict validation using its Qdata")
    args = parser.parse_args()
    started = time.perf_counter()
    if args.mat_cache:
        mat = loadmat(args.mat_cache, variable_names=["BaseEval"], squeeze_me=True, struct_as_record=False)["BaseEval"]
        points = np.asarray(mat.Qdata, dtype=np.float64).T.copy()
        point_source = "MATLAB cached reduced Qdata from the named PLY"
    else:
        points = thin_points(read_points(args.ply), seed=args.seed)
        point_source = "Python radius-thinned PLY"
    result, *arrays = evaluate(args.scan, points, args.dataset_root)
    result.update({"ply": str(args.ply), "ply_sha256": hashlib.sha256(args.ply.read_bytes()).hexdigest(),
                   "point_source": point_source, "runtime_seconds": time.perf_counter() - started})
    if args.mat_cache:
        result["mat_comparison"] = compare_with_mat(result, arrays, args.mat_cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
