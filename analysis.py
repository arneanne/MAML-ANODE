import csv
import json
import math
import os

import numpy as np
from parameter_space_plot import plot_alpha_r_prediction_map

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


SUMMARY_FIELDS = [
    "task",
    "alpha",
    "r",
    "A",
    "loss",
    "bloch_mse",
    "mse_x",
    "mse_y",
    "mse_z",
    "mse_delta",
    "mse_gamma",
    "param_loss",
    "pred_A",
    "pred_alpha",
    "pred_r",
    "err_A",
    "err_alpha",
    "err_r",
    "eta_norm",
    "eta_0",
    "eta_1",
]


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_training_history(run_dir, trainer):
    history = getattr(trainer, "train_history", None) or []
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    if not history:
        return history_path, None

    csv_path = os.path.join(run_dir, "training_history.csv")
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    return history_path, csv_path


def export_test_summary(run_dir, test_results):
    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for task_key in sorted(test_results.keys(), key=lambda item: int(item)):
            res = test_results[task_key]
            eta = np.asarray(res.get("eta", []), dtype=float).reshape(-1)
            row = {
                "task": int(task_key),
                "alpha": res.get("alpha"),
                "r": res.get("r"),
                "A": res.get("A"),
                "loss": res.get("loss"),
                "bloch_mse": res.get("bloch_mse"),
                "mse_x": res.get("mse_x"),
                "mse_y": res.get("mse_y"),
                "mse_z": res.get("mse_z"),
                "mse_delta": res.get("mse_delta"),
                "mse_gamma": res.get("mse_gamma"),
                "param_loss": res.get("param_loss"),
                "pred_A": res.get("pred_A"),
                "pred_alpha": res.get("pred_alpha"),
                "pred_r": res.get("pred_r"),
                "err_A": res.get("err_A"),
                "err_alpha": res.get("err_alpha"),
                "err_r": res.get("err_r"),
                "eta_norm": float(np.linalg.norm(eta)) if eta.size else None,
                "eta_0": float(eta[0]) if eta.size > 0 else None,
                "eta_1": float(eta[1]) if eta.size > 1 else None,
            }
            writer.writerow(row)
    return summary_path


def _mean(values):
    return float(np.mean(values)) if values else None


def _median(values):
    return float(np.median(values)) if values else None


def _quantile(values, q):
    return float(np.quantile(values, q)) if values else None


def _corr(xs, ys):
    if len(xs) < 2 or len(ys) < 2:
        return None
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float)
    if np.allclose(xs_arr, xs_arr[0]) or np.allclose(ys_arr, ys_arr[0]):
        return None
    return float(np.corrcoef(xs_arr, ys_arr)[0, 1])


def _rows_to_numeric(rows, field):
    values = []
    valid_rows = []
    for row in rows:
        value = _safe_float(row.get(field))
        if value is None or math.isnan(value):
            continue
        values.append(value)
        valid_rows.append(row)
    return values, valid_rows


def _top_tasks(rows, metric, reverse=True, top_k=10):
    scored = []
    for row in rows:
        value = _safe_float(row.get(metric))
        if value is None:
            continue
        scored.append((value, row))
    scored.sort(key=lambda item: item[0], reverse=reverse)
    out = []
    for value, row in scored[:top_k]:
        out.append({
            "task": int(row["task"]),
            metric: float(value),
            "alpha": _safe_float(row.get("alpha")),
            "r": _safe_float(row.get("r")),
            "A": _safe_float(row.get("A")),
            "mse_z": _safe_float(row.get("mse_z")),
            "err_alpha": _safe_float(row.get("err_alpha")),
            "err_r": _safe_float(row.get("err_r")),
        })
    return out


def _bin_analysis(rows, group_key, metric_keys, num_bins=3):
    vals, valid_rows = _rows_to_numeric(rows, group_key)
    if len(vals) < num_bins:
        return []
    edges = np.linspace(min(vals), max(vals), num_bins + 1)
    result = []
    for idx in range(num_bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        subset = []
        for row in valid_rows:
            value = _safe_float(row.get(group_key))
            if value is None:
                continue
            if idx == num_bins - 1:
                in_bin = lo <= value <= hi
            else:
                in_bin = lo <= value < hi
            if in_bin:
                subset.append(row)
        if not subset:
            continue
        item = {
            "bin": idx + 1,
            "range": [float(lo), float(hi)],
            "count": len(subset),
        }
        for metric in metric_keys:
            metric_values, _ = _rows_to_numeric(subset, metric)
            item[f"{metric}_mean"] = _mean(metric_values)
        result.append(item)
    return result


def _plot_training_diagnostics(history, out_dir):
    if plt is None:
        return []
    if not history:
        return []
    epochs = [item["epoch"] for item in history]
    saved = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.ravel()
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="train_loss", linewidth=2)
    axes[0].plot(
        epochs,
        [item["val_loss"] if item["val_loss"] is not None else np.nan for item in history],
        label="val_loss",
        linewidth=2,
    )
    axes[0].set_title("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [item["train_recon_loss"] for item in history], label="recon", linewidth=2)
    axes[1].plot(epochs, [item["train_param_loss"] for item in history], label="param", linewidth=2)
    axes[1].plot(epochs, [item["train_eta_reg"] for item in history], label="eta_reg", linewidth=2)
    axes[1].set_title("Loss Components")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, [item["mean_eta_norm"] for item in history], linewidth=2)
    axes[2].set_title("Mean Eta Norm")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("epoch")

    axes[3].plot(epochs, [item["mean_eta_shift"] for item in history], linewidth=2, label="eta_shift")
    axes[3].plot(epochs, [item["lr"] for item in history], linewidth=2, label="lr")
    axes[3].set_title("Adaptation / LR")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].set_xlabel("epoch")

    fig.tight_layout()
    path = os.path.join(out_dir, "training_diagnostics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    return saved


def _plot_test_diagnostics(rows, out_dir):
    saved = []
    if plt is None:
        return saved
    if not rows:
        return saved

    loss = [_safe_float(row.get("loss")) for row in rows]
    bloch = [_safe_float(row.get("bloch_mse")) for row in rows]
    mse_x = [_safe_float(row.get("mse_x")) for row in rows]
    mse_y = [_safe_float(row.get("mse_y")) for row in rows]
    mse_z = [_safe_float(row.get("mse_z")) for row in rows]
    alpha = [_safe_float(row.get("alpha")) for row in rows]
    r = [_safe_float(row.get("r")) for row in rows]
    pred_alpha = [_safe_float(row.get("pred_alpha")) for row in rows]
    pred_r = [_safe_float(row.get("pred_r")) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    axes[0].hist([v for v in loss if v is not None], bins=20, alpha=0.85)
    axes[0].set_title("Loss Distribution")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        [
            [v for v in mse_x if v is not None],
            [v for v in mse_y if v is not None],
            [v for v in mse_z if v is not None],
        ],
        bins=20,
        label=["mse_x", "mse_y", "mse_z"],
        alpha=0.7,
    )
    axes[1].set_title("Component Error Distribution")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].scatter(alpha, pred_alpha, s=18, alpha=0.75)
    diag_alpha = [min(alpha), max(alpha)]
    axes[2].plot(diag_alpha, diag_alpha, linestyle="--", linewidth=1.5, color="black")
    axes[2].set_xlabel("true alpha")
    axes[2].set_ylabel("pred alpha")
    axes[2].set_title("Alpha Calibration")
    axes[2].grid(True, alpha=0.3)

    axes[3].scatter(r, pred_r, s=18, alpha=0.75)
    diag_r = [min(r), max(r)]
    axes[3].plot(diag_r, diag_r, linestyle="--", linewidth=1.5, color="black")
    axes[3].set_xlabel("true r")
    axes[3].set_ylabel("pred r")
    axes[3].set_title("r Calibration")
    axes[3].grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "test_diagnostics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(alpha, r, c=bloch, s=28, cmap="viridis")
    ax.set_xlabel("alpha")
    ax.set_ylabel("r")
    ax.set_title("Bloch MSE over Task Space")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("bloch_mse")
    fig.tight_layout()
    path = os.path.join(out_dir, "task_space_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)
    return saved


def generate_analysis_report(run_dir, trainer, test_results):
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    history = getattr(trainer, "train_history", None) or []
    history_json_path, history_csv_path = export_training_history(run_dir, trainer)
    summary_csv_path = export_test_summary(run_dir, test_results)

    rows = []
    for task_key in sorted(test_results.keys(), key=lambda item: int(item)):
        row = {"task": int(task_key)}
        row.update(test_results[task_key])
        rows.append(row)

    metrics = {}
    for metric in ["loss", "bloch_mse", "mse_x", "mse_y", "mse_z", "mse_delta", "mse_gamma", "err_A", "err_alpha", "err_r"]:
        values, _ = _rows_to_numeric(rows, metric)
        metrics[metric] = {
            "mean": _mean(values),
            "median": _median(values),
            "p90": _quantile(values, 0.9),
            "p95": _quantile(values, 0.95),
            "max": max(values) if values else None,
        }

    z_dominant_count = 0
    for row in rows:
        x_val = _safe_float(row.get("mse_x"))
        y_val = _safe_float(row.get("mse_y"))
        z_val = _safe_float(row.get("mse_z"))
        if x_val is None or y_val is None or z_val is None:
            continue
        if z_val > x_val and z_val > y_val:
            z_dominant_count += 1

    correlations = {}
    for param in ["alpha", "r", "A"]:
        _, valid_rows = _rows_to_numeric(rows, param)
        correlations[param] = {}
        for metric in ["loss", "bloch_mse", "mse_x", "mse_y", "mse_z", "mse_delta", "mse_gamma", "err_alpha", "err_r"]:
            metric_values = []
            aligned_params = []
            for row in valid_rows:
                metric_value = _safe_float(row.get(metric))
                param_value = _safe_float(row.get(param))
                if metric_value is None or param_value is None:
                    continue
                aligned_params.append(param_value)
                metric_values.append(metric_value)
            correlations[param][metric] = _corr(aligned_params, metric_values)

    report = {
        "run_dir": os.path.abspath(run_dir),
        "num_test_tasks": len(rows),
        "num_epochs": len(history),
        "metrics": metrics,
        "z_axis_dominance": {
            "count": z_dominant_count,
            "ratio": float(z_dominant_count / len(rows)) if rows else None,
        },
        "correlations": correlations,
        "hardest_tasks_by_loss": _top_tasks(rows, "loss", reverse=True, top_k=10),
        "easiest_tasks_by_loss": _top_tasks(rows, "loss", reverse=False, top_k=10),
        "bins": {
            "alpha": _bin_analysis(rows, "alpha", ["loss", "bloch_mse", "mse_z", "mse_delta", "mse_gamma", "err_alpha", "err_r"]),
            "r": _bin_analysis(rows, "r", ["loss", "bloch_mse", "mse_z", "mse_delta", "mse_gamma", "err_alpha", "err_r"]),
            "A": _bin_analysis(rows, "A", ["loss", "bloch_mse", "mse_z", "mse_delta", "mse_gamma", "err_alpha", "err_r"]),
        },
        "artifacts": {
            "training_history_json": history_json_path,
            "training_history_csv": history_csv_path,
            "summary_csv": summary_csv_path,
        },
    }

    report["artifacts"]["plots"] = []
    report["artifacts"]["plots"].extend(_plot_training_diagnostics(history, analysis_dir))
    report["artifacts"]["plots"].extend(_plot_test_diagnostics(rows, analysis_dir))
    alpha_r_map_path = plot_alpha_r_prediction_map(
        test_results=test_results,
        train_task_dataset=getattr(trainer, "task_dataset", None),
        out_dir=analysis_dir,
        metric_key="loss",
    )
    if alpha_r_map_path is not None:
        report["artifacts"]["plots"].append(alpha_r_map_path)

    report_path = os.path.join(analysis_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown_path = os.path.join(analysis_dir, "report.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Analysis\n\n")
        f.write(f"- Run dir: `{os.path.abspath(run_dir)}`\n")
        f.write(f"- Test tasks: {len(rows)}\n")
        f.write(f"- Epochs: {len(history)}\n")
        f.write(f"- Z-dominant tasks: {z_dominant_count}/{len(rows) if rows else 0}\n\n")
        f.write("## Key Metrics\n\n")
        for metric, stat in metrics.items():
            mean_text = "nan" if stat["mean"] is None else f"{stat['mean']:.6f}"
            median_text = "nan" if stat["median"] is None else f"{stat['median']:.6f}"
            p90_text = "nan" if stat["p90"] is None else f"{stat['p90']:.6f}"
            p95_text = "nan" if stat["p95"] is None else f"{stat['p95']:.6f}"
            max_text = "nan" if stat["max"] is None else f"{stat['max']:.6f}"
            f.write(
                f"- `{metric}`: mean={mean_text} median={median_text} "
                f"p90={p90_text} p95={p95_text} max={max_text}\n"
            )
        f.write("\n## Hardest Tasks By Loss\n\n")
        for item in report["hardest_tasks_by_loss"][:5]:
            f.write(
                f"- task {item['task']}: loss={item['loss']:.6f}, alpha={item['alpha']:.3f}, "
                f"r={item['r']:.3f}, mse_z={item['mse_z']:.6f}\n"
            )
        f.write("\n## Correlations\n\n")
        for param, corr_map in correlations.items():
            parts = []
            for metric, value in corr_map.items():
                text = "nan" if value is None else f"{value:.3f}"
                parts.append(f"{metric}={text}")
            f.write(f"- `{param}`: " + ", ".join(parts) + "\n")

    return {
        "analysis_dir": analysis_dir,
        "report_json": report_path,
        "report_md": markdown_path,
        "summary_csv": summary_csv_path,
        "training_history_json": history_json_path,
        "training_history_csv": history_csv_path,
    }
