import argparse
import csv
import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch

from data_gen import generate_task_data


def parse_args():
    default_output_dir = os.path.join(os.path.dirname(__file__), "exported_tasks")
    parser = argparse.ArgumentParser(
        description="Export trajectory files for multiple (alpha, r) task combinations."
    )
    parser.add_argument("--output-dir", type=str, default=default_output_dir,
                        help="Directory used to store one file per (alpha, r) task.")
    parser.add_argument("--alpha-min", type=float, default=0.4)
    parser.add_argument("--alpha-max", type=float, default=0.7)
    parser.add_argument("--r-min", type=float, default=0.2)
    parser.add_argument("--r-max", type=float, default=0.5)
    parser.add_argument("--alpha-points", type=int, default=10,
                        help="Number of evenly spaced alpha grid points. Set to 0 to disable grid generation.")
    parser.add_argument("--r-points", type=int, default=10,
                        help="Number of evenly spaced r grid points. Set to 0 to disable grid generation.")
    parser.add_argument("--num-random-points", type=int, default=0,
                        help="Number of extra random (alpha, r) samples drawn uniformly from the interval.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-traj", type=int, default=2000)
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--omega0", type=float, default=1.0)
    parser.add_argument("--M", type=float, default=0.4)
    parser.add_argument("--zeta", type=float, default=0.9)
    parser.add_argument("--kBT", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing task files if they already exist.")
    return parser.parse_args()


def validate_args(args):
    if args.alpha_min >= args.alpha_max:
        raise ValueError("alpha_min must be smaller than alpha_max")
    if args.r_min >= args.r_max:
        raise ValueError("r_min must be smaller than r_max")
    if args.alpha_points < 0 or args.r_points < 0:
        raise ValueError("alpha_points and r_points must be >= 0")
    if args.num_random_points < 0:
        raise ValueError("num_random_points must be >= 0")
    if args.alpha_points == 1 or args.r_points == 1:
        raise ValueError("alpha_points and r_points should be 0 or >= 2")
    if args.alpha_points == 0 and args.r_points != 0:
        raise ValueError("r_points must also be 0 when alpha_points is 0")
    if args.r_points == 0 and args.alpha_points != 0:
        raise ValueError("alpha_points must also be 0 when r_points is 0")
    if args.num_traj <= 0:
        raise ValueError("num_traj must be > 0")
    if args.T <= 0 or args.dt <= 0:
        raise ValueError("T and dt must be > 0")


def build_task_points(args) -> List[Tuple[str, float, float]]:
    points: List[Tuple[str, float, float]] = []
    seen = set()

    if args.alpha_points > 0 and args.r_points > 0:
        alpha_grid = np.linspace(args.alpha_min, args.alpha_max, args.alpha_points)
        r_grid = np.linspace(args.r_min, args.r_max, args.r_points)
        for alpha in alpha_grid:
            for r in r_grid:
                key = (round(float(alpha), 10), round(float(r), 10))
                if key not in seen:
                    seen.add(key)
                    points.append(("grid", float(alpha), float(r)))

    rng = np.random.default_rng(args.seed)
    attempts = 0
    while sum(1 for source, _, _ in points if source == "random") < args.num_random_points:
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        r = float(rng.uniform(args.r_min, args.r_max))
        key = (round(alpha, 10), round(r, 10))
        attempts += 1
        if key in seen:
            if attempts > max(100, args.num_random_points * 20):
                raise RuntimeError("Failed to generate enough unique random (alpha, r) pairs.")
            continue
        seen.add(key)
        points.append(("random", alpha, r))

    return points


def format_value_for_name(value: float) -> str:
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def build_file_name(alpha: float, r: float) -> str:
    return f"task_alpha_{format_value_for_name(alpha)}__r_{format_value_for_name(r)}.pt"


def main():
    args = parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)

    task_points = build_task_points(args)
    if not task_points:
        raise ValueError("No task points were generated. Configure grid points and/or random points.")

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

    print("\nExport completed.")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Total task files: {len(manifest)}")


if __name__ == "__main__":
    main()
