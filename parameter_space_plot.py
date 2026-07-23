import os

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:  # pragma: no cover - optional dependency
    plt = None
    Rectangle = None


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_train_bounds(train_task_dataset):
    if not train_task_dataset:
        return None

    alpha_vals = []
    r_vals = []
    for task_data in train_task_dataset.values():
        alpha = _safe_float(task_data.get("alpha"))
        radius = _safe_float(task_data.get("r"))
        if alpha is None or radius is None:
            continue
        alpha_vals.append(alpha)
        r_vals.append(radius)

    if not alpha_vals or not r_vals:
        return None

    return {
        "alpha_min": float(min(alpha_vals)),
        "alpha_max": float(max(alpha_vals)),
        "r_min": float(min(r_vals)),
        "r_max": float(max(r_vals)),
    }


def _collect_plot_rows(test_results):
    rows = []
    for task_key in sorted(test_results.keys(), key=lambda item: int(item)):
        res = test_results[task_key]
        row = {
            "task": int(task_key),
            "alpha": _safe_float(res.get("alpha")),
            "r": _safe_float(res.get("r")),
            "pred_alpha": _safe_float(res.get("pred_alpha")),
            "pred_r": _safe_float(res.get("pred_r")),
            "loss": _safe_float(res.get("loss")),
        }
        if None in (row["alpha"], row["r"], row["pred_alpha"], row["pred_r"], row["loss"]):
            continue
        rows.append(row)
    return rows


def _compute_axis_limits(rows, train_bounds):
    alpha_values = []
    r_values = []
    for row in rows:
        alpha_values.extend([row["alpha"], row["pred_alpha"]])
        r_values.extend([row["r"], row["pred_r"]])

    if train_bounds is not None:
        alpha_values.extend([train_bounds["alpha_min"], train_bounds["alpha_max"]])
        r_values.extend([train_bounds["r_min"], train_bounds["r_max"]])

    if not alpha_values or not r_values:
        return None

    alpha_min = min(alpha_values)
    alpha_max = max(alpha_values)
    r_min = min(r_values)
    r_max = max(r_values)

    alpha_pad = max(0.03, 0.08 * max(alpha_max - alpha_min, 1e-6))
    r_pad = max(0.03, 0.08 * max(r_max - r_min, 1e-6))
    return {
        "xlim": (alpha_min - alpha_pad, alpha_max + alpha_pad),
        "ylim": (r_min - r_pad, r_max + r_pad),
    }


def plot_alpha_r_prediction_map(test_results, train_task_dataset, out_dir, metric_key="loss"):
    if plt is None:
        return None

    rows = _collect_plot_rows(test_results)
    if not rows:
        return None

    train_bounds = _extract_train_bounds(train_task_dataset)
    limits = _compute_axis_limits(rows, train_bounds)
    if limits is None:
        return None

    true_alpha = np.asarray([row["alpha"] for row in rows], dtype=float)
    true_r = np.asarray([row["r"] for row in rows], dtype=float)
    pred_alpha = np.asarray([row["pred_alpha"] for row in rows], dtype=float)
    pred_r = np.asarray([row["pred_r"] for row in rows], dtype=float)
    metric_values = np.asarray([row[metric_key] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 7.2))

    # Draw task-wise displacement from the true location to the predicted location.
    for idx in range(len(rows)):
        ax.plot(
            [true_alpha[idx], pred_alpha[idx]],
            [true_r[idx], pred_r[idx]],
            color="0.65",
            linewidth=0.9,
            alpha=0.55,
            zorder=1,
        )

    true_scatter = ax.scatter(
        true_alpha,
        true_r,
        s=48,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        alpha=0.95,
        label="true (alpha, r)",
        zorder=3,
    )
    pred_scatter = ax.scatter(
        pred_alpha,
        pred_r,
        c=metric_values,
        cmap="viridis",
        s=54,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
        label=f"predicted (colored by {metric_key})",
        zorder=4,
    )

    if train_bounds is not None and Rectangle is not None:
        rect = Rectangle(
            (train_bounds["alpha_min"], train_bounds["r_min"]),
            train_bounds["alpha_max"] - train_bounds["alpha_min"],
            train_bounds["r_max"] - train_bounds["r_min"],
            fill=False,
            linestyle="--",
            linewidth=1.8,
            edgecolor="tab:red",
            label="train ID boundary",
            zorder=2,
        )
        ax.add_patch(rect)

    colorbar = fig.colorbar(pred_scatter, ax=ax)
    colorbar.set_label(metric_key)

    ax.set_xlim(*limits["xlim"])
    ax.set_ylim(*limits["ylim"])
    ax.set_xlabel("true alpha")
    ax.set_ylabel("true r")
    ax.set_title("Alpha-r Prediction Map")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    path = os.path.join(out_dir, "alpha_r_prediction_map.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
