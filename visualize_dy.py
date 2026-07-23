import argparse
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


def _require_torch_serialization() -> None:
    if hasattr(torch, "load") and hasattr(torch, "save"):
        return
    torch_file = getattr(torch, "__file__", None)
    raise RuntimeError(
        "当前 Python 环境没有可用的 PyTorch 序列化接口，无法读取 .pt 任务文件。"
        f" 当前导入到的 torch={torch!r}, torch.__file__={torch_file!r}。"
        " 请切换到安装了完整 PyTorch 的环境，例如项目训练使用的 conda 环境。"
    )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _tensor_to_numpy_safe(tensor: Any) -> np.ndarray:
    return np.asarray(tensor.detach().cpu().tolist(), dtype=float)


def _load_manifest(task_dir: str) -> Optional[Dict[str, Any]]:
    manifest_path = os.path.join(task_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _discover_task_records(task_dir: str) -> List[Dict[str, Any]]:
    manifest = _load_manifest(task_dir)
    if manifest is not None:
        records = []
        for item in manifest.get("tasks", []):
            file_path = item.get("file_path")
            file_name = item.get("file_name")
            resolved_path = None
            if file_path and os.path.isfile(file_path):
                resolved_path = os.path.abspath(file_path)
            elif file_name:
                candidate = os.path.join(task_dir, file_name)
                if os.path.isfile(candidate):
                    resolved_path = os.path.abspath(candidate)
            if resolved_path is None:
                continue
            records.append(
                {
                    "file_path": resolved_path,
                    "file_name": os.path.basename(resolved_path),
                    "alpha": _safe_float(item.get("alpha")),
                    "r": _safe_float(item.get("r")),
                }
            )
        if records:
            return records

    records = []
    for name in sorted(os.listdir(task_dir)):
        if name.endswith(".pt") and name.startswith("task_alpha_"):
            records.append(
                {
                    "file_path": os.path.abspath(os.path.join(task_dir, name)),
                    "file_name": name,
                    "alpha": None,
                    "r": None,
                }
            )
    return records


def _discover_task_files(task_dir: str) -> List[str]:
    return [item["file_path"] for item in _discover_task_records(task_dir)]


def _select_task_files(
    input_path: str,
    max_tasks: int,
    seed: int,
    file_pattern: Optional[str],
) -> List[str]:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"input path not found: {input_path}")

    file_paths = _discover_task_files(input_path)
    if file_pattern:
        file_paths = [path for path in file_paths if file_pattern in os.path.basename(path)]
    if not file_paths:
        raise FileNotFoundError(f"no task .pt files found under: {input_path}")

    if max_tasks > 0 and len(file_paths) > max_tasks:
        rng = random.Random(seed)
        file_paths = sorted(rng.sample(file_paths, max_tasks))
    return file_paths


def _prepare_out_dir(out_dir: Optional[str], input_path: str) -> str:
    if out_dir is not None:
        target = os.path.abspath(out_dir)
    else:
        base = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
        target = os.path.join(os.path.abspath(base), "dy_visualizations")
    os.makedirs(target, exist_ok=True)
    return target


def _choose_default_anchor(values: Sequence[float]) -> Optional[float]:
    unique = sorted({round(float(value), 12) for value in values if value is not None})
    if not unique:
        return None
    return float(unique[len(unique) // 2])


def _format_float_token(value: float) -> str:
    return f"{float(value):.6f}".replace(".", "p")


def _select_trajectory_indices(num_traj: int, max_traj: int, seed: int) -> List[int]:
    if num_traj <= 0:
        return []
    if max_traj <= 0 or max_traj >= num_traj:
        return list(range(num_traj))
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_traj), max_traj))


def _build_heatmap_data(dy: np.ndarray, max_heatmap_traj: int) -> np.ndarray:
    if dy.shape[0] <= max_heatmap_traj:
        return dy
    selected = np.linspace(0, dy.shape[0] - 1, max_heatmap_traj, dtype=int)
    return dy[selected]


def _extract_task_plot_stats(task_data: Dict[str, Any], task_file: str) -> Dict[str, Any]:
    if "dY" not in task_data or "t" not in task_data:
        raise KeyError(f"{task_file} does not contain required keys 'dY' and 't'")

    dy = _tensor_to_numpy_safe(task_data["dY"])
    t = _tensor_to_numpy_safe(task_data["t"])
    if dy.ndim != 2:
        raise ValueError(f"{task_file} has unexpected dY shape {dy.shape}; expected [num_traj, num_time]")
    if t.ndim != 1:
        raise ValueError(f"{task_file} has unexpected t shape {t.shape}; expected [num_time]")
    if dy.shape[1] != t.shape[0]:
        raise ValueError(f"{task_file} has mismatched dY/t lengths: {dy.shape} vs {t.shape}")

    return {
        "task_file": os.path.abspath(task_file),
        "task_name": os.path.splitext(os.path.basename(task_file))[0],
        "alpha": _safe_float(task_data.get("alpha")),
        "r": _safe_float(task_data.get("r")),
        "dy": dy,
        "t": t,
        "num_traj": int(dy.shape[0]),
        "num_time": int(dy.shape[1]),
        "dy_mean": dy.mean(axis=0),
        "dy_std": dy.std(axis=0),
        "dy_q10": np.quantile(dy, 0.10, axis=0),
        "dy_q50": np.quantile(dy, 0.50, axis=0),
        "dy_q90": np.quantile(dy, 0.90, axis=0),
    }


def _plot_task_dy(
    task_data: Dict[str, Any],
    task_file: str,
    out_dir: str,
    max_traj: int,
    max_heatmap_traj: int,
    alpha_overlay: float,
    seed: int,
) -> str:
    if plt is None:
        raise RuntimeError("matplotlib is required for dY visualization")
    stats = _extract_task_plot_stats(task_data=task_data, task_file=task_file)
    dy = stats["dy"]
    t = stats["t"]
    alpha = stats["alpha"]
    radius = stats["r"]
    num_traj = stats["num_traj"]
    num_time = stats["num_time"]

    show_indices = _select_trajectory_indices(num_traj=num_traj, max_traj=max_traj, seed=seed)
    dy_show = dy[show_indices] if show_indices else np.empty((0, num_time), dtype=float)
    dy_mean = stats["dy_mean"]
    dy_std = stats["dy_std"]
    dy_heatmap = _build_heatmap_data(dy, max_heatmap_traj=max_heatmap_traj)
    dy_flat = dy.reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_overlay, ax_stats, ax_heatmap, ax_hist = axes.ravel()

    for idx, traj_idx in enumerate(show_indices):
        ax_overlay.plot(
            t,
            dy_show[idx],
            linewidth=1.0,
            alpha=alpha_overlay,
            label=f"traj {traj_idx}" if idx < 8 else None,
        )
    ax_overlay.set_title("dY Trajectories")
    ax_overlay.set_xlabel("t")
    ax_overlay.set_ylabel("dY")
    ax_overlay.grid(True, alpha=0.25)
    if show_indices and len(show_indices) <= 8:
        ax_overlay.legend(loc="best", fontsize=8)

    ax_stats.plot(t, dy_mean, color="tab:blue", linewidth=2.0, label="mean dY")
    ax_stats.fill_between(
        t,
        dy_mean - dy_std,
        dy_mean + dy_std,
        color="tab:blue",
        alpha=0.2,
        label="mean ± std",
    )
    ax_stats.set_title("Mean and Std")
    ax_stats.set_xlabel("t")
    ax_stats.set_ylabel("dY")
    ax_stats.grid(True, alpha=0.25)
    ax_stats.legend(loc="best")

    im = ax_heatmap.imshow(
        dy_heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(t[0]), float(t[-1]), 0, dy_heatmap.shape[0]],
        cmap="viridis",
    )
    ax_heatmap.set_title("dY Heatmap")
    ax_heatmap.set_xlabel("t")
    ax_heatmap.set_ylabel("trajectory index")
    fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04, label="dY")

    ax_hist.hist(dy_flat, bins=50, color="tab:green", alpha=0.85, edgecolor="black", linewidth=0.3)
    ax_hist.axvline(float(dy_flat.mean()), color="tab:red", linestyle="--", linewidth=1.8, label="mean")
    ax_hist.set_title("dY Value Distribution")
    ax_hist.set_xlabel("dY")
    ax_hist.set_ylabel("count")
    ax_hist.grid(True, alpha=0.2)
    ax_hist.legend(loc="best")

    title_parts = [os.path.basename(task_file)]
    if alpha is not None and radius is not None:
        title_parts.append(f"alpha={alpha:.4f}, r={radius:.4f}")
    title_parts.append(f"num_traj={num_traj}, num_time={num_time}")
    fig.suptitle(" | ".join(title_parts), fontsize=13)

    out_name = os.path.splitext(os.path.basename(task_file))[0] + "__dy.png"
    out_path = os.path.join(out_dir, out_name)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_multi_task_mean_comparison(task_stats: Sequence[Dict[str, Any]], out_dir: str) -> Optional[str]:
    if plt is None or len(task_stats) < 2:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for item in task_stats:
        label_parts = [item["task_name"]]
        if item["alpha"] is not None and item["r"] is not None:
            label_parts.append(f"a={item['alpha']:.3f}, r={item['r']:.3f}")
        ax.plot(
            item["t"],
            item["dy_mean"],
            linewidth=2.0,
            alpha=0.95,
            label=" | ".join(label_parts),
        )

    ax.set_title("Mean dY Comparison Across Tasks")
    ax.set_xlabel("t")
    ax.set_ylabel("mean dY")
    ax.grid(True, alpha=0.25)
    if len(task_stats) <= 8:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)

    out_path = os.path.join(out_dir, "dy_mean_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_multi_task_mean_std_comparison(task_stats: Sequence[Dict[str, Any]], out_dir: str) -> Optional[str]:
    if plt is None or len(task_stats) < 2:
        return None

    fig, axes = plt.subplots(len(task_stats), 1, figsize=(11, max(3.0 * len(task_stats), 6.5)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, item in zip(axes, task_stats):
        label = item["task_name"]
        if item["alpha"] is not None and item["r"] is not None:
            label += f" | a={item['alpha']:.3f}, r={item['r']:.3f}"
        ax.plot(item["t"], item["dy_mean"], linewidth=2.0, color="tab:blue", label="mean dY")
        ax.fill_between(
            item["t"],
            item["dy_mean"] - item["dy_std"],
            item["dy_mean"] + item["dy_std"],
            color="tab:blue",
            alpha=0.2,
            label="mean ± std",
        )
        ax.set_ylabel("dY")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("t")
    fig.suptitle("Multi-task Mean ± Std Comparison", fontsize=13)
    out_path = os.path.join(out_dir, "dy_mean_std_comparison.png")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_multi_task_quantile_comparison(task_stats: Sequence[Dict[str, Any]], out_dir: str) -> Optional[str]:
    if plt is None or len(task_stats) < 2:
        return None

    fig, axes = plt.subplots(len(task_stats), 1, figsize=(11, max(3.0 * len(task_stats), 6.5)), sharex=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, item in zip(axes, task_stats):
        label = item["task_name"]
        if item["alpha"] is not None and item["r"] is not None:
            label += f" | a={item['alpha']:.3f}, r={item['r']:.3f}"
        ax.plot(item["t"], item["dy_q50"], linewidth=2.0, color="tab:purple", label="q50")
        ax.fill_between(
            item["t"],
            item["dy_q10"],
            item["dy_q90"],
            color="tab:purple",
            alpha=0.18,
            label="q10 - q90",
        )
        ax.plot(item["t"], item["dy_q10"], linewidth=1.0, color="tab:purple", alpha=0.8, linestyle="--")
        ax.plot(item["t"], item["dy_q90"], linewidth=1.0, color="tab:purple", alpha=0.8, linestyle="--")
        ax.set_ylabel("dY")
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("t")
    fig.suptitle("Multi-task Quantile Band Comparison", fontsize=13)
    out_path = os.path.join(out_dir, "dy_quantile_comparison.png")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _select_sweep_records(
    task_dir: str,
    fixed_alpha: Optional[float],
    fixed_r: Optional[float],
    atol: float = 1e-10,
) -> Dict[str, Any]:
    records = _discover_task_records(task_dir)
    valid_records = [item for item in records if item["alpha"] is not None and item["r"] is not None]
    if not valid_records:
        return {
            "fixed_alpha_value": None,
            "fixed_r_value": None,
            "fixed_alpha_records": [],
            "fixed_r_records": [],
        }

    alpha_values = [float(item["alpha"]) for item in valid_records]
    r_values = [float(item["r"]) for item in valid_records]
    alpha_anchor = _choose_default_anchor(alpha_values) if fixed_alpha is None else float(fixed_alpha)
    r_anchor = _choose_default_anchor(r_values) if fixed_r is None else float(fixed_r)

    fixed_alpha_records = [
        item for item in valid_records
        if abs(float(item["alpha"]) - alpha_anchor) <= atol
    ]
    fixed_r_records = [
        item for item in valid_records
        if abs(float(item["r"]) - r_anchor) <= atol
    ]
    fixed_alpha_records.sort(key=lambda item: float(item["r"]))
    fixed_r_records.sort(key=lambda item: float(item["alpha"]))
    return {
        "fixed_alpha_value": alpha_anchor,
        "fixed_r_value": r_anchor,
        "fixed_alpha_records": fixed_alpha_records,
        "fixed_r_records": fixed_r_records,
    }


def _load_task_stats_for_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    task_stats = []
    for item in records:
        task_data = torch.load(item["file_path"], map_location="cpu")
        task_stats.append(_extract_task_plot_stats(task_data=task_data, task_file=item["file_path"]))
    return task_stats


def _plot_parameter_sweep_mean_comparison(
    task_stats: Sequence[Dict[str, Any]],
    out_dir: str,
    title: str,
    label_key: str,
    fixed_key: str,
    fixed_value: float,
    file_name: str,
) -> Optional[str]:
    if plt is None or len(task_stats) < 2:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for item in task_stats:
        varying_value = item[label_key]
        label = f"{label_key}={varying_value:.3f}" if varying_value is not None else item["task_name"]
        ax.plot(item["t"], item["dy_mean"], linewidth=2.0, alpha=0.95, label=label)

    ax.set_title(f"{title} | fixed {fixed_key}={fixed_value:.3f}")
    ax.set_xlabel("t")
    ax.set_ylabel("mean dY")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    out_path = os.path.join(out_dir, file_name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Visualize weak-measurement dY trajectories from exported task files")
    parser.add_argument(
        "input_path",
        type=str,
        help="单个任务 .pt 文件，或包含多个 task .pt 文件的目录",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录；默认写到输入目录下的 dy_visualizations",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=4,
        help="当输入是目录时，最多抽样可视化多少个任务；<=0 表示全部",
    )
    parser.add_argument(
        "--max-traj",
        type=int,
        default=24,
        help="每个任务叠加显示多少条轨迹；<=0 表示全部",
    )
    parser.add_argument(
        "--max-heatmap-traj",
        type=int,
        default=128,
        help="热力图最多显示多少条轨迹",
    )
    parser.add_argument(
        "--file-pattern",
        type=str,
        default=None,
        help="仅可视化文件名包含该子串的任务文件",
    )
    parser.add_argument(
        "--alpha-overlay",
        type=float,
        default=0.35,
        help="轨迹叠加图的透明度",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="目录抽样任务或轨迹时使用的随机种子",
    )
    parser.add_argument(
        "--fixed-alpha",
        type=float,
        default=None,
        help="当输入是规则网格目录时，额外生成一张固定 alpha、r 从低到高变化的 dY_mean 对比图",
    )
    parser.add_argument(
        "--fixed-r",
        type=float,
        default=None,
        help="当输入是规则网格目录时，额外生成一张固定 r、alpha 从低到高变化的 dY_mean 对比图",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _require_torch_serialization()
    task_files = _select_task_files(
        input_path=args.input_path,
        max_tasks=args.max_tasks,
        seed=args.seed,
        file_pattern=args.file_pattern,
    )
    out_dir = _prepare_out_dir(out_dir=args.out_dir, input_path=args.input_path)

    manifest = []
    comparison_items = []
    for idx, task_file in enumerate(task_files):
        task_data = torch.load(task_file, map_location="cpu")
        task_stats = _extract_task_plot_stats(task_data=task_data, task_file=task_file)
        out_path = _plot_task_dy(
            task_data=task_data,
            task_file=task_file,
            out_dir=out_dir,
            max_traj=args.max_traj,
            max_heatmap_traj=args.max_heatmap_traj,
            alpha_overlay=args.alpha_overlay,
            seed=args.seed + idx,
        )
        comparison_items.append(task_stats)
        item = {
            "task_file": os.path.abspath(task_file),
            "figure_path": os.path.abspath(out_path),
            "alpha": task_stats["alpha"],
            "r": task_stats["r"],
            "num_traj": task_stats["num_traj"],
            "num_time": task_stats["num_time"],
        }
        manifest.append(item)
        print(f"[{idx + 1}/{len(task_files)}] saved {out_path}")

    comparison_path = _plot_multi_task_mean_comparison(comparison_items, out_dir=out_dir)
    mean_std_path = _plot_multi_task_mean_std_comparison(comparison_items, out_dir=out_dir)
    quantile_path = _plot_multi_task_quantile_comparison(comparison_items, out_dir=out_dir)
    for figure_path in [comparison_path, mean_std_path, quantile_path]:
        if figure_path is not None:
            print(f"saved comparison figure {figure_path}")

    fixed_alpha_path = None
    fixed_r_path = None
    fixed_alpha_value = None
    fixed_r_value = None
    if os.path.isdir(args.input_path):
        sweep_info = _select_sweep_records(
            task_dir=os.path.abspath(args.input_path),
            fixed_alpha=args.fixed_alpha,
            fixed_r=args.fixed_r,
        )
        fixed_alpha_value = sweep_info["fixed_alpha_value"]
        fixed_r_value = sweep_info["fixed_r_value"]
        if len(sweep_info["fixed_alpha_records"]) >= 2 and fixed_alpha_value is not None:
            fixed_alpha_stats = _load_task_stats_for_records(sweep_info["fixed_alpha_records"])
            fixed_alpha_path = _plot_parameter_sweep_mean_comparison(
                task_stats=fixed_alpha_stats,
                out_dir=out_dir,
                title="Mean dY Sweep Along r",
                label_key="r",
                fixed_key="alpha",
                fixed_value=fixed_alpha_value,
                file_name=f"dy_mean_fixed_alpha_{_format_float_token(fixed_alpha_value)}.png",
            )
        if len(sweep_info["fixed_r_records"]) >= 2 and fixed_r_value is not None:
            fixed_r_stats = _load_task_stats_for_records(sweep_info["fixed_r_records"])
            fixed_r_path = _plot_parameter_sweep_mean_comparison(
                task_stats=fixed_r_stats,
                out_dir=out_dir,
                title="Mean dY Sweep Along alpha",
                label_key="alpha",
                fixed_key="r",
                fixed_value=fixed_r_value,
                file_name=f"dy_mean_fixed_r_{_format_float_token(fixed_r_value)}.png",
            )
        for figure_path in [fixed_alpha_path, fixed_r_path]:
            if figure_path is not None:
                print(f"saved sweep figure {figure_path}")

    manifest_path = os.path.join(out_dir, "dy_visualization_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_path": os.path.abspath(args.input_path),
                "output_dir": os.path.abspath(out_dir),
                "num_tasks_visualized": len(manifest),
                "comparison_figure": os.path.abspath(comparison_path) if comparison_path is not None else None,
                "mean_std_comparison_figure": os.path.abspath(mean_std_path) if mean_std_path is not None else None,
                "quantile_comparison_figure": os.path.abspath(quantile_path) if quantile_path is not None else None,
                "fixed_alpha_value": fixed_alpha_value,
                "fixed_r_value": fixed_r_value,
                "fixed_alpha_sweep_figure": os.path.abspath(fixed_alpha_path) if fixed_alpha_path is not None else None,
                "fixed_r_sweep_figure": os.path.abspath(fixed_r_path) if fixed_r_path is not None else None,
                "items": manifest,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
