"""Evaluate the 22 DTU test scans with the selected single-scan evaluator."""

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


DTU_SCANS = (1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34,
             48, 49, 62, 75, 77, 110, 114, 118)
METRICS = ("accuracy_mean_mm", "completeness_mean_mm", "overall_mean_mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True,
                        help="Directory containing scan1.ply, scan4.ply, etc.")
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="DTU MVS Data directory containing ObsMask and Points")
    parser.add_argument("--output-dir", type=Path,
                        help="Per-scan JSON and summary directory (default: DATA/eval)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "torch"), default="cpu",
                        help="Single-scan implementation (default: cpu)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device when --backend torch (default: cuda)")
    parser.add_argument("--scans", type=int, nargs="+",
                        help="Optional subset for a pilot run; default: all 22 scans")
    args = parser.parse_args()

    scans = args.scans if args.scans is not None else DTU_SCANS
    if len(set(scans)) != len(scans) or not set(scans) <= set(DTU_SCANS):
        parser.error("--scans must contain unique DTU test scan numbers")
    inputs = {scan: args.data / f"scan{scan}.ply" for scan in scans}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        parser.error("missing reconstruction PLYs:\n" + "\n".join(missing))
    if not args.dataset_dir.is_dir():
        parser.error(f"DTU data directory does not exist: {args.dataset_dir}")

    output_dir = args.output_dir or args.data / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = Path(__file__).with_name(
        "dtu_eval_torch.py" if args.backend == "torch" else "dtu_eval_python.py")
    results = []
    for scan in scans:
        output = output_dir / f"scan{scan}.json"
        command = [sys.executable, str(evaluator), "--scan", str(scan),
                   "--ply", str(inputs[scan]), "--dataset-root", str(args.dataset_dir),
                   "--output", str(output), "--seed", str(args.seed)]
        if args.backend == "torch":
            command += ["--device", args.device]
        subprocess.run(command, check=True)
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("scan") != scan or not all(
                isinstance(result.get(key), (int, float)) and math.isfinite(result[key])
                for key in METRICS):
            raise ValueError(f"Invalid evaluator result: {output}")
        results.append(result)

    summary = {
        "scans": list(scans),
        "complete_test_set": set(scans) == set(DTU_SCANS),
        "seed": args.seed,
        "backend": args.backend,
        "device": args.device if args.backend == "torch" else "cpu",
        "mean_accuracy_mm": statistics.fmean(r["accuracy_mean_mm"] for r in results),
        "mean_completeness_mm": statistics.fmean(r["completeness_mean_mm"] for r in results),
        "mean_overall_mm": statistics.fmean(r["overall_mean_mm"] for r in results),
        "per_scan": [{"scan": r["scan"], **{key: r[key] for key in METRICS},
                      "ply_sha256": r["ply_sha256"]} for r in results],
    }
    suffix = "all22" if summary["complete_test_set"] else "subset_" + "_".join(map(str, scans))
    summary_path = output_dir / f"summary_{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **summary}), flush=True)


if __name__ == "__main__":
    main()
