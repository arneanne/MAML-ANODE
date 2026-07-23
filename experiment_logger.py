import json
import os
from datetime import datetime

import numpy as np


def _now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _jsonify(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(item) for item in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _format_value(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_duration(seconds):
    if seconds is None:
        return "n/a"
    total_seconds = float(seconds)
    if total_seconds < 60.0:
        return f"{total_seconds:.2f}s"
    minutes, sec = divmod(total_seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {sec:.2f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"


class ExperimentLogger:
    def __init__(self, run_dir):
        self.run_dir = os.path.abspath(run_dir)
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "experiment_log.jsonl")
        self.md_path = os.path.join(self.run_dir, "experiment_log.md")

        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Log\n\n")
            f.write(f"- Run dir: `{self.run_dir}`\n")
            f.write(f"- Started at: {_now_iso()}\n\n")

    def log_event(self, step, title, details=None, payload=None):
        entry = {
            "timestamp": _now_iso(),
            "step": step,
            "title": title,
            "details": details or [],
            "payload": _jsonify(payload),
        }

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"## {step}. {title}\n\n")
            f.write(f"- Time: `{entry['timestamp']}`\n")
            for item in entry["details"]:
                f.write(f"- {item}\n")
            if payload:
                f.write("\n```json\n")
                f.write(json.dumps(entry["payload"], ensure_ascii=False, indent=2))
                f.write("\n```\n")
            f.write("\n")

    def log_config(self, args_dict, default_dict, run_name, run_dir, save_dir):
        changed = {}
        for key, value in sorted(args_dict.items()):
            if default_dict.get(key) != value:
                changed[key] = {
                    "default": default_dict.get(key),
                    "current": value,
                }

        details = [
            f"Run name: `{run_name}`",
            f"Save dir: `{os.path.abspath(save_dir)}`",
            f"Changed params: {len(changed)}",
        ]
        if changed:
            changed_brief = ", ".join(
                f"`{key}`={_format_value(meta['current'])}"
                for key, meta in changed.items()
            )
            details.append(f"Adjusted params: {changed_brief}")

        payload = {
            "run_name": run_name,
            "run_dir": os.path.abspath(run_dir),
            "save_dir": os.path.abspath(save_dir),
            "changed_params": changed,
            "full_config": _jsonify(args_dict),
        }
        self.log_event("1", "Run Configuration", details, payload)

    def log_dataset_summary(
        self,
        train_params,
        test_params,
        train_ids,
        val_ids,
        tasks_per_epoch_effective,
        num_time_steps,
        random_window_enabled,
    ):
        def _range(values):
            if not values:
                return None
            return [float(min(values)), float(max(values))]

        train_alpha = [float(alpha) for alpha, _ in train_params]
        train_r = [float(radius) for _, radius in train_params]
        test_alpha = [float(alpha) for alpha, _ in test_params]
        test_r = [float(radius) for _, radius in test_params]

        payload = {
            "train_task_count_loaded": len(train_params),
            "test_task_count_loaded": len(test_params),
            "train_ids": list(train_ids),
            "val_ids": list(val_ids),
            "tasks_per_epoch_effective": int(tasks_per_epoch_effective),
            "num_time_steps": int(num_time_steps),
            "random_window_enabled": bool(random_window_enabled),
            "train_alpha_range": _range(train_alpha),
            "train_r_range": _range(train_r),
            "test_alpha_range": _range(test_alpha),
            "test_r_range": _range(test_r),
        }
        details = [
            f"Loaded train/test tasks: {len(train_params)} / {len(test_params)}",
            f"Train/val split: {len(train_ids)} / {len(val_ids)}",
            f"Train alpha range: {payload['train_alpha_range']}",
            f"Train r range: {payload['train_r_range']}",
            f"Test alpha range: {payload['test_alpha_range']}",
            f"Test r range: {payload['test_r_range']}",
            f"Time steps: {num_time_steps}, random windows: {'enabled' if random_window_enabled else 'disabled'}",
        ]
        self.log_event("2", "Dataset Summary", details, payload)

    def log_training_summary(self, trainer):
        history = getattr(trainer, "train_history", None) or []
        if not history:
            self.log_event("3", "Training Summary", ["No training history recorded."], {})
            return

        first = history[0]
        last = history[-1]
        best_train = min(history, key=lambda item: item.get("train_loss", float("inf")))
        val_items = [item for item in history if item.get("val_loss") is not None]
        best_val = min(val_items, key=lambda item: item["val_loss"]) if val_items else None

        details = [
            f"Epochs completed: {len(history)}",
            f"Train loss: {first.get('train_loss', float('nan')):.6f} -> {last.get('train_loss', float('nan')):.6f}",
            f"Train recon loss: {last.get('train_recon_loss', float('nan')):.6f}",
            f"Train param loss: {last.get('train_param_loss', float('nan')):.6f}",
            f"Mean eta norm / shift: {last.get('mean_eta_norm', float('nan')):.6f} / {last.get('mean_eta_shift', float('nan')):.6f}",
        ]
        if best_val is not None:
            details.append(
                f"Best val loss: {best_val['val_loss']:.6f} at epoch {best_val['epoch']}"
            )

        payload = {
            "epochs": len(history),
            "first_epoch": _jsonify(first),
            "last_epoch": _jsonify(last),
            "best_train_epoch": _jsonify(best_train),
            "best_val_epoch": _jsonify(best_val),
        }
        self.log_event("3", "Training Summary", details, payload)

    def log_test_summary(self, test_results):
        rows = []
        for task_key in sorted(test_results.keys(), key=lambda item: int(item)):
            res = test_results[task_key]
            rows.append({
                "task": int(task_key),
                "loss": _safe_float(res.get("loss")),
                "bloch_mse": _safe_float(res.get("bloch_mse")),
                "mse_x": _safe_float(res.get("mse_x")),
                "mse_y": _safe_float(res.get("mse_y")),
                "mse_z": _safe_float(res.get("mse_z")),
                "mse_delta": _safe_float(res.get("mse_delta")),
                "mse_gamma": _safe_float(res.get("mse_gamma")),
                "err_alpha": _safe_float(res.get("err_alpha")),
                "err_r": _safe_float(res.get("err_r")),
                "alpha": _safe_float(res.get("alpha")),
                "r": _safe_float(res.get("r")),
            })

        losses = [row["loss"] for row in rows if row["loss"] is not None]
        bloch = [row["bloch_mse"] for row in rows if row["bloch_mse"] is not None]
        mse_x = [row["mse_x"] for row in rows if row["mse_x"] is not None]
        mse_y = [row["mse_y"] for row in rows if row["mse_y"] is not None]
        mse_z = [row["mse_z"] for row in rows if row["mse_z"] is not None]
        mse_delta = [row["mse_delta"] for row in rows if row["mse_delta"] is not None]
        mse_gamma = [row["mse_gamma"] for row in rows if row["mse_gamma"] is not None]
        err_alpha = [row["err_alpha"] for row in rows if row["err_alpha"] is not None]
        err_r = [row["err_r"] for row in rows if row["err_r"] is not None]

        z_dominant = 0
        for row in rows:
            if None in (row["mse_x"], row["mse_y"], row["mse_z"]):
                continue
            if row["mse_z"] > row["mse_x"] and row["mse_z"] > row["mse_y"]:
                z_dominant += 1

        hardest = sorted(
            [row for row in rows if row["loss"] is not None],
            key=lambda item: item["loss"],
            reverse=True,
        )[:5]

        details = [
            f"Test tasks evaluated: {len(rows)}",
            f"Mean loss / Bloch MSE: {np.mean(losses):.6f} / {np.mean(bloch):.6f}" if losses and bloch else "Mean loss / Bloch MSE: n/a",
            (
                f"Mean x/y/z MSE: {np.mean(mse_x):.6f} / {np.mean(mse_y):.6f} / {np.mean(mse_z):.6f}"
                if mse_x and mse_y and mse_z else "Mean x/y/z MSE: n/a"
            ),
            f"Mean z MSE: {np.mean(mse_z):.6f}" if mse_z else "Mean z MSE: n/a",
            (
                f"Mean delta/gamma MSE: {np.mean(mse_delta):.6f} / {np.mean(mse_gamma):.6f}"
                if mse_delta and mse_gamma else "Mean delta/gamma MSE: n/a"
            ),
            f"Mean |alpha error| / |r error|: {np.mean(err_alpha):.6f} / {np.mean(err_r):.6f}" if err_alpha and err_r else "Mean |alpha error| / |r error|: n/a",
            f"Z-dominant tasks: {z_dominant}/{len(rows)}",
        ]
        payload = {
            "num_test_tasks": len(rows),
            "mean_loss": float(np.mean(losses)) if losses else None,
            "mean_bloch_mse": float(np.mean(bloch)) if bloch else None,
            "mean_mse_x": float(np.mean(mse_x)) if mse_x else None,
            "mean_mse_y": float(np.mean(mse_y)) if mse_y else None,
            "mean_mse_z": float(np.mean(mse_z)) if mse_z else None,
            "mean_mse_delta": float(np.mean(mse_delta)) if mse_delta else None,
            "mean_mse_gamma": float(np.mean(mse_gamma)) if mse_gamma else None,
            "mean_err_alpha": float(np.mean(err_alpha)) if err_alpha else None,
            "mean_err_r": float(np.mean(err_r)) if err_r else None,
            "z_dominant_tasks": z_dominant,
            "hardest_tasks_by_loss": hardest,
        }
        self.log_event("4", "Test Summary", details, payload)

    def log_artifacts(self, artifact_paths):
        details = [f"Generated: `{os.path.abspath(path)}`" for path in artifact_paths if path]
        payload = {
            "artifacts": [os.path.abspath(path) for path in artifact_paths if path],
        }
        self.log_event("5", "Artifacts", details, payload)

    def log_timing_summary(self, run_total_sec, train_time_sec, test_time_sec, per_epoch_times_sec):
        per_epoch_times = [float(item) for item in (per_epoch_times_sec or [])]
        avg_epoch = float(np.mean(per_epoch_times)) if per_epoch_times else None
        min_epoch = float(np.min(per_epoch_times)) if per_epoch_times else None
        max_epoch = float(np.max(per_epoch_times)) if per_epoch_times else None

        details = [
            f"Run total time: {_format_duration(run_total_sec)}",
            f"Train time: {_format_duration(train_time_sec)}",
            f"Test time: {_format_duration(test_time_sec)}",
            (
                "Per-epoch time: "
                f"avg={_format_duration(avg_epoch)}, "
                f"min={_format_duration(min_epoch)}, "
                f"max={_format_duration(max_epoch)}"
            ) if per_epoch_times else "Per-epoch time: n/a",
        ]
        payload = {
            "run_total_sec": _safe_float(run_total_sec),
            "train_time_sec": _safe_float(train_time_sec),
            "test_time_sec": _safe_float(test_time_sec),
            "per_epoch_times_sec": per_epoch_times,
            "per_epoch_avg_sec": _safe_float(avg_epoch),
            "per_epoch_min_sec": _safe_float(min_epoch),
            "per_epoch_max_sec": _safe_float(max_epoch),
        }
        self.log_event("6", "Timing Summary", details, payload)
