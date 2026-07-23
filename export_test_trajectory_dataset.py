import argparse
import csv
import json
import os
import random

import numpy as np
import torch

from data_gen import generate_task_data
from export_train_trajectory_dataset import build_file_name, build_task_points, validate_args


def parse_args():
    default_output_dir = os.path.join(os.path.dirname(__file__), "exported_test_tasks")
    parser = argparse.ArgumentParser(
        description="Export test trajectory files for multiple (alpha, r) task combinations."
    )
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Directory used to store one file per test (alpha, r) task.")
    parser.add_argument("--alpha-min", type=float, default=0.2)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--r-min", type=float, default=0.1)
    parser.add_argument("--r-max", type=float, default=0.6)
    parser.add_argument("--alpha-points", type=int, default=0,
                        help="Number of evenly spaced alpha grid points. Default disables grid generation.")
    parser.add_argument("--r-points", type=int, default=0,
                        help="Number of evenly spaced r grid points. Default disables grid generation.")
    parser.add_argument("--num-random-points", type=int, default=100,
                        help="Number of random test (alpha, r) samples drawn uniformly from the interval.")
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--num-traj", type=int, default=256)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--M", type=float, default=0.4)
    parser.add_argument("--zeta", type=float, default=0.9)
    parser.add_argument("--kBT", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing task files if they already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    task_points = build_task_points(args)
    if not task_points:
        raise ValueError("No test task points were generated. Configure grid points and/or random points.")

    manifest = []
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for idx, (source, alpha, r) in enumerate(task_points):
        file_name = build_file_name(alpha, r)
        file_path = os.path.join(args.output_dir, file_name)
        if os.path.exists(file_path) and not args.overwrite:
            raise FileExistsError(
                f"File already exists: {file_path}. Use --overwrite to replace existing files."
            )

        task_seed = args.seed + idx
        task_data = generate_task_data(
            alpha=alpha,
            r=r,
            num_traj=args.num_traj,
            T=args.T,
            dt=args.dt,
            omega0=args.omega0,
            M=args.M,
            zeta=args.zeta,
            kBT=args.kBT,
            seed=task_seed,
        )
        torch.save(task_data, file_path)

        item = {
            "task_index": idx,
            "source": source,
            "alpha": alpha,
            "r": r,
            "num_traj": args.num_traj,
            "T": args.T,
            "dt": args.dt,
            "omega0": args.omega0,
            "M": args.M,
            "zeta": args.zeta,
            "kBT": args.kBT,
            "seed": task_seed,
            "file_name": file_name,
            "file_path": os.path.abspath(file_path),
        }
        manifest.append(item)
        print(
            f"[{idx + 1}/{len(task_points)}] {source:<6} alpha={alpha:.6f} r={r:.6f} -> {file_name}"
        )

    manifest_json = {
        "output_dir": os.path.abspath(args.output_dir),
        "total_tasks": len(manifest),
        "grid_tasks": sum(1 for item in manifest if item["source"] == "grid"),
        "random_tasks": sum(1 for item in manifest if item["source"] == "random"),
        "config": vars(args),
        "tasks": manifest,
    }
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest_json, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.output_dir, "manifest.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_index", "source", "alpha", "r", "num_traj", "T", "dt",
                "omega0", "M", "zeta", "kBT", "seed", "file_name", "file_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print("\nTest export completed.")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Total test task files: {len(manifest)}")


if __name__ == "__main__":
    main()
