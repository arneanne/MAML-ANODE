import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency
    pd = None


ROOT_MARKER_FILES = {
    "config.json",
    "task_split.json",
    "summary.csv",
    "training_history.csv",
    "test_results.json",
    "experiment_log.jsonl",
    "experiment_log.md",
    "launcher_spec.json",
}

STRUCTURED_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet"}
SUMMARY_NUMERIC_FIELDS = {
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
}
DEFAULT_METRICS = [
    "mean_loss",
    "mean_bloch_mse",
    "mean_err_alpha",
    "mean_err_r",
    "z_dominant_ratio",
    "best_val_loss",
    "final_train_loss",
]
DEFAULT_GROUP_BY = [
    "param__outer_lr",
    "param__inner_lr",
    "param__w_z",
    "param__tasks_per_epoch",
]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_int(value: Any) -> Optional[int]:
    number = _safe_float(value)
    if number is None:
        return None
    return int(number)


def _safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return slug.strip("_") or "value"


def _flatten_dict(obj: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in (obj or {}).items():
        new_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=f"{new_key}__"))
        else:
            flat[new_key] = value
    return flat


def _json_dump(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid jsonl line {line_no}: {exc}") from exc
    return rows


def _read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _read_parquet(path: str) -> List[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("pandas is required to read parquet files")
    frame = pd.read_parquet(path)
    return frame.to_dict(orient="records")


def _structured_loader(path: str) -> Tuple[str, Any]:
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return "csv", _read_csv(path)
    if suffix == ".json":
        return "json", _read_json(path)
    if suffix == ".jsonl":
        return "jsonl", _read_jsonl(path)
    if suffix == ".parquet":
        return "parquet", _read_parquet(path)
    raise ValueError(f"unsupported structured suffix: {suffix}")


def _infer_row_count(payload: Any) -> Optional[int]:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        if "tasks" in payload and isinstance(payload["tasks"], list):
            return len(payload["tasks"])
        return len(payload)
    return None


def _mean(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    return float(np.median(values)) if values else None


def _std(values: Sequence[float]) -> Optional[float]:
    return float(np.std(values, ddof=0)) if values else None


def _variance(values: Sequence[float]) -> Optional[float]:
    return float(np.var(values, ddof=0)) if values else None


def _quantile(values: Sequence[float], q: float) -> Optional[float]:
    return float(np.quantile(values, q)) if values else None


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    return None


def _collect_numeric(rows: Sequence[Dict[str, Any]], field: str) -> List[float]:
    values = []
    for row in rows:
        number = _safe_float(row.get(field))
        if number is not None:
            values.append(number)
    return values


def _first_existing(path_candidates: Sequence[str]) -> Optional[str]:
    for path in path_candidates:
        if os.path.exists(path):
            return path
    return None


def _natural_run_key(run_name: str) -> Tuple[int, str]:
    match = re.search(r"(\d+)", run_name)
    if match:
        return int(match.group(1)), run_name
    return 10**9, run_name


def _parse_range_spec(range_spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if not range_spec:
        return None
    match = re.fullmatch(r"\s*(\d+)\s*[:-]\s*(\d+)\s*", range_spec)
    if not match:
        raise ValueError("--run-range 必须形如 1:20 或 1-20")
    start = int(match.group(1))
    end = int(match.group(2))
    if start > end:
        start, end = end, start
    return start, end


def _match_run_filters(
    run_name: str,
    run_dir: str,
    include_pattern: Optional[str],
    exclude_pattern: Optional[str],
    run_range: Optional[Tuple[int, int]],
) -> bool:
    target = f"{run_name} {run_dir}"
    if include_pattern and re.search(include_pattern, target) is None:
        return False
    if exclude_pattern and re.search(exclude_pattern, target) is not None:
        return False
    if run_range is not None:
        match = re.search(r"(\d+)", run_name)
        if match is None:
            return False
        run_id = int(match.group(1))
        if not (run_range[0] <= run_id <= run_range[1]):
            return False
    return True


def find_scan_roots(project_root: str, explicit_roots: Optional[Sequence[str]]) -> List[str]:
    if explicit_roots:
        roots = [os.path.abspath(path) for path in explicit_roots]
    else:
        candidates = [
            os.path.join(project_root, "results"),
            os.path.join(project_root, "results_managed"),
        ]
        roots = [path for path in candidates if os.path.isdir(path)]
    unique = []
    seen = set()
    for path in roots:
        norm = os.path.abspath(path)
        if norm not in seen:
            unique.append(norm)
            seen.add(norm)
    return unique


def _is_run_dir(dirpath: str, dirnames: Sequence[str], filenames: Sequence[str]) -> bool:
    filename_set = set(filenames)
    if ROOT_MARKER_FILES & filename_set:
        return True
    if "analysis" in dirnames and os.path.isfile(os.path.join(dirpath, "analysis", "report.json")):
        return True
    if (
        "meta_aqnode_results" in dirnames
        and os.path.isfile(os.path.join(dirpath, "meta_aqnode_results", "analysis_summary.json"))
    ):
        return True
    return False


def discover_run_dirs(
    roots: Sequence[str],
    include_pattern: Optional[str],
    exclude_pattern: Optional[str],
    run_range: Optional[Tuple[int, int]],
) -> List[str]:
    discovered = []
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            if _is_run_dir(dirpath, dirnames, filenames):
                run_dir = os.path.abspath(dirpath)
                run_name = os.path.basename(run_dir)
                if _match_run_filters(run_name, run_dir, include_pattern, exclude_pattern, run_range):
                    if run_dir not in seen:
                        discovered.append(run_dir)
                        seen.add(run_dir)
                dirnames[:] = []
    discovered.sort(key=lambda path: _natural_run_key(os.path.basename(path)))
    return discovered


def _summary_from_test_results(test_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for task_key, res in sorted(test_results.items(), key=lambda item: int(item[0])):
        row = {"task": int(task_key)}
        for key, value in res.items():
            if isinstance(value, (list, dict)):
                continue
            row[key] = value
        rows.append(row)
    return rows


def _resolve_run_timestamp(run_dir: str, experiment_events: Sequence[Dict[str, Any]]) -> Tuple[str, float]:
    for event in experiment_events:
        ts = _parse_iso_datetime(event.get("timestamp"))
        if ts is not None:
            return ts.isoformat(timespec="seconds"), float(ts.timestamp())
    mtime = os.path.getmtime(run_dir)
    dt = datetime.fromtimestamp(mtime)
    return dt.isoformat(timespec="seconds"), float(mtime)


def _collect_structured_inventory(run_dir: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    inventory = []
    issues = []
    for dirpath, _, filenames in os.walk(run_dir):
        for name in sorted(filenames):
            suffix = os.path.splitext(name)[1].lower()
            if suffix not in STRUCTURED_EXTENSIONS:
                continue
            path = os.path.join(dirpath, name)
            relative_path = os.path.relpath(path, run_dir)
            item = {
                "relative_path": relative_path,
                "absolute_path": path,
                "suffix": suffix,
                "status": "ok",
                "row_count": None,
                "error": None,
            }
            try:
                _, payload = _structured_loader(path)
                item["row_count"] = _infer_row_count(payload)
            except Exception as exc:  # pragma: no cover - defensive
                item["status"] = "error"
                item["error"] = str(exc)
                issues.append(f"{relative_path}: {exc}")
            inventory.append(item)
    return inventory, issues


def _load_run_artifacts(run_dir: str) -> Dict[str, Any]:
    issues: List[str] = []
    payload: Dict[str, Any] = {
        "run_dir": run_dir,
        "run_name": os.path.basename(run_dir),
        "issues": issues,
    }

    known_paths = {
        "config": os.path.join(run_dir, "config.json"),
        "task_split": os.path.join(run_dir, "task_split.json"),
        "summary": os.path.join(run_dir, "summary.csv"),
        "training_history_csv": os.path.join(run_dir, "training_history.csv"),
        "training_history_json": os.path.join(run_dir, "training_history.json"),
        "test_results": os.path.join(run_dir, "test_results.json"),
        "report": os.path.join(run_dir, "analysis", "report.json"),
        "experiment_log": os.path.join(run_dir, "experiment_log.jsonl"),
        "launcher_spec": os.path.join(run_dir, "launcher_spec.json"),
    }
    payload["paths"] = known_paths

    inventory, inventory_issues = _collect_structured_inventory(run_dir)
    payload["structured_inventory"] = inventory
    issues.extend(inventory_issues)

    if os.path.isfile(known_paths["config"]):
        try:
            payload["config"] = _read_json(known_paths["config"])
        except Exception as exc:
            issues.append(f"config.json 读取失败: {exc}")
    if os.path.isfile(known_paths["task_split"]):
        try:
            payload["task_split"] = _read_json(known_paths["task_split"])
        except Exception as exc:
            issues.append(f"task_split.json 读取失败: {exc}")
    if os.path.isfile(known_paths["summary"]):
        try:
            payload["summary_rows"] = _read_csv(known_paths["summary"])
        except Exception as exc:
            issues.append(f"summary.csv 读取失败: {exc}")
    if os.path.isfile(known_paths["training_history_csv"]):
        try:
            payload["history_rows"] = _read_csv(known_paths["training_history_csv"])
        except Exception as exc:
            issues.append(f"training_history.csv 读取失败: {exc}")
    elif os.path.isfile(known_paths["training_history_json"]):
        try:
            payload["history_rows"] = _read_json(known_paths["training_history_json"])
        except Exception as exc:
            issues.append(f"training_history.json 读取失败: {exc}")
    if os.path.isfile(known_paths["test_results"]):
        try:
            payload["test_results"] = _read_json(known_paths["test_results"])
        except Exception as exc:
            issues.append(f"test_results.json 读取失败: {exc}")
    if os.path.isfile(known_paths["report"]):
        try:
            payload["report"] = _read_json(known_paths["report"])
        except Exception as exc:
            issues.append(f"analysis/report.json 读取失败: {exc}")
    if os.path.isfile(known_paths["experiment_log"]):
        try:
            payload["experiment_events"] = _read_jsonl(known_paths["experiment_log"])
        except Exception as exc:
            issues.append(f"experiment_log.jsonl 读取失败: {exc}")
    if os.path.isfile(known_paths["launcher_spec"]):
        try:
            payload["launcher_spec"] = _read_json(known_paths["launcher_spec"])
        except Exception as exc:
            issues.append(f"launcher_spec.json 读取失败: {exc}")

    if "summary_rows" not in payload and "test_results" in payload:
        payload["summary_rows"] = _summary_from_test_results(payload["test_results"])

    if "experiment_events" not in payload:
        payload["experiment_events"] = []
    created_at, created_at_ts = _resolve_run_timestamp(run_dir, payload["experiment_events"])
    payload["created_at"] = created_at
    payload["created_at_ts"] = created_at_ts
    return payload


def _validate_run_payload(run_payload: Dict[str, Any]) -> str:
    issues = run_payload["issues"]
    summary_rows = run_payload.get("summary_rows") or []
    history_rows = run_payload.get("history_rows") or []
    report = run_payload.get("report") or {}
    test_results = run_payload.get("test_results") or {}

    if not summary_rows and not report and not test_results:
        issues.append("缺少 summary.csv / test_results.json / analysis/report.json，无法做核心指标分析")
    if not history_rows:
        issues.append("缺少 training_history.csv 或 training_history.json，无法做训练趋势分析")

    if summary_rows and test_results:
        if len(summary_rows) != len(test_results):
            issues.append(
                f"summary 行数与 test_results 任务数不一致: {len(summary_rows)} vs {len(test_results)}"
            )
    if summary_rows and report:
        report_tasks = _safe_int(report.get("num_test_tasks"))
        if report_tasks is not None and report_tasks != len(summary_rows):
            issues.append(f"summary 行数与 report.num_test_tasks 不一致: {len(summary_rows)} vs {report_tasks}")

    has_fatal_error = any("读取失败" in issue or "无法做核心指标分析" in issue for issue in issues)
    if has_fatal_error:
        return "invalid"
    if issues:
        return "partial"
    return "ok"


def _extract_config_fields(run_payload: Dict[str, Any]) -> Dict[str, Any]:
    config = run_payload.get("config") or {}
    launcher_spec = run_payload.get("launcher_spec") or {}
    flat = {}
    flat.update(_flatten_dict(config.get("user_config") or {}, prefix="param__"))
    flat.update(_flatten_dict(config.get("effective_config") or {}, prefix="config__"))
    if launcher_spec:
        flat.update(_flatten_dict(launcher_spec, prefix="launcher__"))
    task_split = run_payload.get("task_split") or {}
    if task_split:
        flat.update(_flatten_dict(task_split, prefix="split__"))
    return flat


def _compute_summary_metrics(summary_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    metrics["summary_row_count"] = len(summary_rows)
    if not summary_rows:
        return metrics

    for field in sorted(SUMMARY_NUMERIC_FIELDS):
        values = _collect_numeric(summary_rows, field)
        if not values:
            continue
        metrics[f"mean_{field}"] = _mean(values)
        metrics[f"median_{field}"] = _median(values)
        metrics[f"var_{field}"] = _variance(values)
        metrics[f"min_{field}"] = min(values)
        metrics[f"max_{field}"] = max(values)

    z_dominant_count = 0
    total = 0
    for row in summary_rows:
        x = _safe_float(row.get("mse_x"))
        y = _safe_float(row.get("mse_y"))
        z = _safe_float(row.get("mse_z"))
        if None in (x, y, z):
            continue
        total += 1
        if z > x and z > y:
            z_dominant_count += 1
    metrics["z_total_tasks"] = total
    metrics["z_dominant_count"] = z_dominant_count
    metrics["z_dominant_ratio"] = (float(z_dominant_count) / float(total)) if total else None
    metrics["num_tasks"] = len(summary_rows)
    return metrics


def _compute_history_metrics(history_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "history_row_count": len(history_rows),
        "epochs_recorded": len(history_rows),
    }
    if not history_rows:
        return metrics

    def _best_row(key: str) -> Optional[Dict[str, Any]]:
        candidates = []
        for row in history_rows:
            value = _safe_float(row.get(key))
            if value is not None:
                candidates.append((value, row))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    last = history_rows[-1]
    metrics["final_train_loss"] = _safe_float(last.get("train_loss"))
    metrics["final_val_loss"] = _safe_float(last.get("val_loss"))
    metrics["final_mean_eta_norm"] = _safe_float(last.get("mean_eta_norm"))
    metrics["final_mean_eta_shift"] = _safe_float(last.get("mean_eta_shift"))
    metrics["final_inner_lr_alpha"] = _safe_float(last.get("inner_lr_alpha"))
    metrics["final_inner_lr_r"] = _safe_float(last.get("inner_lr_r"))
    metrics["best_selection_score"] = None
    metrics["best_val_loss"] = None

    best_val = _best_row("val_loss")
    if best_val is not None:
        metrics["best_val_loss"] = _safe_float(best_val.get("val_loss"))
        metrics["best_selection_score"] = _safe_float(best_val.get("selection_score"))

    train_losses = _collect_numeric(history_rows, "train_loss")
    if train_losses:
        metrics["best_task_loss"] = min(train_losses)
    return metrics


def build_catalog_entry(run_payload: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "run_name": run_payload["run_name"],
        "run_dir": run_payload["run_dir"],
        "created_at": run_payload["created_at"],
        "created_at_ts": run_payload["created_at_ts"],
    }
    entry.update(_extract_config_fields(run_payload))
    entry.update(_compute_summary_metrics(run_payload.get("summary_rows") or []))
    entry.update(_compute_history_metrics(run_payload.get("history_rows") or []))
    entry["issue_count"] = len(run_payload["issues"])
    entry["issue_summary"] = " | ".join(run_payload["issues"])
    entry["data_status"] = run_payload["data_status"]
    return entry


def _metric_candidates(metrics: Sequence[str], rows: Sequence[Dict[str, Any]]) -> List[str]:
    available = []
    for metric in metrics:
        if any(_safe_float(row.get(metric)) is not None for row in rows):
            available.append(metric)
    return available


def _normalize_metric_names(raw_metrics: Sequence[str]) -> List[str]:
    normalized = []
    for metric in raw_metrics:
        metric = metric.strip()
        if not metric:
            continue
        if metric in SUMMARY_NUMERIC_FIELDS:
            normalized.append(f"mean_{metric}")
        else:
            normalized.append(metric)
    return normalized


def _descriptive_stats(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    result = {}
    for metric in metrics:
        values = _collect_numeric(rows, metric)
        if not values:
            continue
        result[metric] = {
            "count": len(values),
            "mean": _mean(values),
            "median": _median(values),
            "std": _std(values),
            "variance": _variance(values),
            "min": min(values),
            "max": max(values),
            "q25": _quantile(values, 0.25),
            "q75": _quantile(values, 0.75),
        }
    return result


def _group_analysis(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str], metrics: Sequence[str]) -> Dict[str, Any]:
    analysis = {}
    for group_field in group_fields:
        groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            value = row.get(group_field)
            if value in (None, "", "None"):
                continue
            groups[str(value)].append(row)
        if len(groups) < 2:
            continue

        field_payload = {"groups": {}, "metric_gaps": {}}
        for group_name, group_rows in sorted(groups.items()):
            field_payload["groups"][group_name] = {
                "count": len(group_rows),
                "metrics": _descriptive_stats(group_rows, metrics),
            }

        for metric in metrics:
            group_means = []
            for group_name, group_rows in groups.items():
                values = _collect_numeric(group_rows, metric)
                if values:
                    group_means.append((group_name, float(np.mean(values))))
            if len(group_means) < 2:
                continue
            group_means.sort(key=lambda item: item[1])
            field_payload["metric_gaps"][metric] = {
                "best_group": {"name": group_means[0][0], "mean": group_means[0][1]},
                "worst_group": {"name": group_means[-1][0], "mean": group_means[-1][1]},
                "gap": group_means[-1][1] - group_means[0][1],
            }
        analysis[group_field] = field_payload
    return analysis


def _trend_analysis(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda item: (item.get("created_at_ts", 0.0), _natural_run_key(item["run_name"])))
    run_order = list(range(len(ordered)))
    result = {}
    for metric in metrics:
        values = []
        valid_order = []
        for idx, row in enumerate(ordered):
            number = _safe_float(row.get(metric))
            if number is None:
                continue
            valid_order.append(run_order[idx])
            values.append(number)
        if len(values) < 2:
            continue
        result[metric] = {
            "run_order_correlation": _pearson(valid_order, values),
            "start_value": values[0],
            "end_value": values[-1],
        }
    return result


def _correlation_analysis(rows: Sequence[Dict[str, Any]], metrics: Sequence[str]) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[Dict[str, Any]]]:
    numeric_fields = []
    for key in sorted({field for row in rows for field in row.keys()}):
        values = _collect_numeric(rows, key)
        if len(values) >= 2:
            numeric_fields.append(key)

    selected_fields = list(dict.fromkeys(list(metrics) + [field for field in numeric_fields if field.startswith("param__")]))
    selected_fields = [field for field in selected_fields if field in numeric_fields]
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    pairs = []
    for left in selected_fields:
        matrix[left] = {}
        left_values = []
        for row in rows:
            left_values.append(_safe_float(row.get(left)))
        for right in selected_fields:
            aligned_left = []
            aligned_right = []
            for row in rows:
                lv = _safe_float(row.get(left))
                rv = _safe_float(row.get(right))
                if lv is None or rv is None:
                    continue
                aligned_left.append(lv)
                aligned_right.append(rv)
            corr = _pearson(aligned_left, aligned_right)
            matrix[left][right] = corr
            if corr is not None:
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "pearson": corr,
                        "count": len(aligned_left),
                    }
                )
    unique_pairs = []
    seen = set()
    for item in pairs:
        key = tuple(sorted((item["left"], item["right"])))
        if key in seen:
            continue
        seen.add(key)
        unique_pairs.append(item)
    unique_pairs.sort(key=lambda item: abs(item["pearson"]), reverse=True)
    return matrix, unique_pairs[:20]


def _write_catalog_csv(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    all_keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_bar(rows: Sequence[Dict[str, Any]], metric: str, out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    labels = []
    values = []
    for row in rows:
        value = _safe_float(row.get(metric))
        if value is None:
            continue
        labels.append(row["run_name"])
        values.append(value)
    if not values:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5.5))
    ax.bar(labels, values, color="tab:blue", alpha=0.85)
    ax.set_title(f"Run Comparison: {metric}")
    ax.set_xlabel("Run")
    ax.set_ylabel(metric)
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    path = os.path.join(out_dir, f"bar_{_safe_slug(metric)}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_box(rows: Sequence[Dict[str, Any]], metrics: Sequence[str], out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    series = []
    labels = []
    for metric in metrics:
        values = _collect_numeric(rows, metric)
        if not values:
            continue
        series.append(values)
        labels.append(metric)
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(labels)), 5.5))
    ax.boxplot(series, tick_labels=labels, patch_artist=True)
    ax.set_title("Metric Distribution Across Runs")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    path = os.path.join(out_dir, "box_metrics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_heatmap(matrix: Dict[str, Dict[str, Optional[float]]], out_dir: str) -> Optional[str]:
    if plt is None or not matrix:
        return None
    labels = list(matrix.keys())
    if len(labels) < 2:
        return None
    data = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for i, left in enumerate(labels):
        for j, right in enumerate(labels):
            value = matrix[left].get(right)
            if value is not None:
                data[i, j] = value

    fig, ax = plt.subplots(figsize=(max(7, 0.6 * len(labels)), max(6, 0.5 * len(labels))))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Correlation Heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r")
    path = os.path.join(out_dir, "heatmap_correlations.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_trend(rows: Sequence[Dict[str, Any]], metric: str, out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    ordered = sorted(rows, key=lambda item: (item.get("created_at_ts", 0.0), _natural_run_key(item["run_name"])))
    labels = []
    values = []
    for row in ordered:
        value = _safe_float(row.get(metric))
        if value is None:
            continue
        labels.append(row["run_name"])
        values.append(value)
    if len(values) < 2:
        return None

    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5.5))
    ax.plot(labels, values, marker="o", linewidth=2.0, color="tab:green")
    ax.set_title(f"Trend: {metric}")
    ax.set_xlabel("Run Order")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    path = os.path.join(out_dir, f"trend_{_safe_slug(metric)}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_scatter(rows: Sequence[Dict[str, Any]], x_metric: str, y_metric: str, out_dir: str) -> Optional[str]:
    if plt is None:
        return None
    xs = []
    ys = []
    labels = []
    for row in rows:
        x_val = _safe_float(row.get(x_metric))
        y_val = _safe_float(row.get(y_metric))
        if x_val is None or y_val is None:
            continue
        xs.append(x_val)
        ys.append(y_val)
        labels.append(row["run_name"])
    if len(xs) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 5.5))
    ax.scatter(xs, ys, s=60, alpha=0.85, edgecolors="black", linewidths=0.4)
    for label, x_val, y_val in zip(labels, xs, ys):
        ax.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_title(f"{x_metric} vs {y_metric}")
    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.grid(True, alpha=0.3)
    corr = _pearson(xs, ys)
    if corr is not None:
        ax.text(
            0.02,
            0.98,
            f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
    path = os.path.join(out_dir, f"scatter_{_safe_slug(x_metric)}_vs_{_safe_slug(y_metric)}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_group_bars(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str], metric: str, out_dir: str) -> List[str]:
    if plt is None:
        return []
    saved = []
    for group_field in group_fields:
        groups: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            group_value = row.get(group_field)
            metric_value = _safe_float(row.get(metric))
            if group_value in (None, "", "None") or metric_value is None:
                continue
            groups[str(group_value)].append(metric_value)
        if len(groups) < 2:
            continue

        labels = sorted(groups.keys(), key=lambda item: item)
        means = [float(np.mean(groups[label])) for label in labels]
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(labels)), 5.5))
        ax.bar(labels, means, color="tab:orange", alpha=0.85)
        ax.set_title(f"Group Mean: {metric} by {group_field}")
        ax.set_xlabel(group_field)
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=35)
        for idx, value in enumerate(means):
            ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
        path = os.path.join(out_dir, f"group_{_safe_slug(group_field)}__{_safe_slug(metric)}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)
    return saved


def generate_plots(rows: Sequence[Dict[str, Any]], metrics: Sequence[str], group_fields: Sequence[str], out_dir: str) -> List[str]:
    saved = []
    metric_candidates = _metric_candidates(metrics, rows)
    if not metric_candidates:
        return saved

    primary_metric = metric_candidates[0]
    plotters = [
        _plot_bar(rows, primary_metric, out_dir),
        _plot_box(rows, metric_candidates[: min(5, len(metric_candidates))], out_dir),
        _plot_trend(rows, primary_metric, out_dir),
    ]
    saved.extend([path for path in plotters if path])

    numeric_params = []
    for field in sorted({key for row in rows for key in row.keys() if key.startswith("param__")}):
        values = _collect_numeric(rows, field)
        if len(set(round(value, 12) for value in values)) >= 2:
            numeric_params.append(field)

    ranked_params = []
    for field in numeric_params:
        aligned_x = []
        aligned_y = []
        for row in rows:
            x_val = _safe_float(row.get(field))
            y_val = _safe_float(row.get(primary_metric))
            if x_val is None or y_val is None:
                continue
            aligned_x.append(x_val)
            aligned_y.append(y_val)
        corr = _pearson(aligned_x, aligned_y)
        ranked_params.append((0.0 if corr is None else abs(corr), field))
    ranked_params.sort(reverse=True)

    for _, field in ranked_params[:3]:
        path = _plot_scatter(rows, field, primary_metric, out_dir)
        if path:
            saved.append(path)

    corr_matrix, _ = _correlation_analysis(rows, metric_candidates)
    heatmap = _plot_heatmap(corr_matrix, out_dir)
    if heatmap:
        saved.append(heatmap)

    saved.extend(_plot_group_bars(rows, group_fields, primary_metric, out_dir))
    return saved


def write_markdown_report(path: str, report: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Aggregate Experiment Report\n\n")
        f.write(f"- Generated at: `{report['generated_at']}`\n")
        f.write(f"- Roots: `{', '.join(report['roots'])}`\n")
        f.write(f"- Output dir: `{report['output_dir']}`\n")
        f.write(f"- Total scanned runs: `{report['total_scanned_runs']}`\n")
        f.write(f"- Valid runs: `{report['valid_runs']}`\n")
        f.write(f"- Invalid runs: `{report['invalid_runs']}`\n\n")

        f.write("## Descriptive Stats\n\n")
        for metric, stats in report.get("descriptive_stats", {}).items():
            f.write(f"### {metric}\n\n")
            for key, value in stats.items():
                f.write(f"- {key}: `{value}`\n")
            f.write("\n")

        if report.get("group_analysis"):
            f.write("## Group Analysis\n\n")
            for group_field, payload in report["group_analysis"].items():
                f.write(f"### {group_field}\n\n")
                for metric, gap in payload.get("metric_gaps", {}).items():
                    f.write(
                        f"- {metric}: best=`{gap['best_group']['name']}` ({gap['best_group']['mean']:.6g}), "
                        f"worst=`{gap['worst_group']['name']}` ({gap['worst_group']['mean']:.6g}), "
                        f"gap=`{gap['gap']:.6g}`\n"
                    )
                f.write("\n")

        if report.get("trend_analysis"):
            f.write("## Trend Analysis\n\n")
            for metric, trend in report["trend_analysis"].items():
                f.write(
                    f"- {metric}: corr=`{trend['run_order_correlation']}`, "
                    f"start=`{trend['start_value']}`, end=`{trend['end_value']}`\n"
                )
            f.write("\n")

        if report.get("top_correlations"):
            f.write("## Top Correlations\n\n")
            for item in report["top_correlations"]:
                f.write(
                    f"- {item['left']} vs {item['right']}: "
                    f"pearson=`{item['pearson']:.6g}`, count=`{item['count']}`\n"
                )
            f.write("\n")

        if report.get("plots"):
            f.write("## Plots\n\n")
            for path_item in report["plots"]:
                f.write(f"- `{path_item}`\n")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Analyze experiment runs recursively and generate aggregate statistics")
    parser.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="需要扫描的根目录列表；默认自动使用项目下的 results 和 results_managed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="聚合分析输出目录；默认写到 <project>/run_analysis",
    )
    parser.add_argument(
        "--include-pattern",
        type=str,
        default=None,
        help="仅分析 run_name 或路径匹配该正则的目录",
    )
    parser.add_argument(
        "--exclude-pattern",
        type=str,
        default=None,
        help="跳过 run_name 或路径匹配该正则的目录",
    )
    parser.add_argument(
        "--run-range",
        type=str,
        default=None,
        help="仅分析某个 run id 范围，例如 1:20 或 3-15",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="要纳入聚合统计与作图的指标，支持 loss/bloch_mse/err_alpha 等基础名，也支持 mean_loss 这类聚合名",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default=",".join(DEFAULT_GROUP_BY),
        help="组间分析字段，多个字段用逗号分隔，例如 param__outer_lr,param__w_z",
    )
    parser.add_argument(
        "--include-partial",
        action="store_true",
        help="将 partial 状态的 run 也纳入统计；默认只统计 data_status=ok 的 run",
    )
    parser.add_argument(
        "--limit-runs",
        type=int,
        default=0,
        help="仅分析前 N 个 run；0 表示不限制",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    roots = find_scan_roots(project_root, args.roots)
    if not roots:
        raise SystemExit("未找到可扫描的结果根目录，请通过 --roots 显式指定。")

    output_dir = os.path.abspath(args.output_dir or os.path.join(project_root, "run_analysis"))
    os.makedirs(output_dir, exist_ok=True)

    run_range = _parse_range_spec(args.run_range)
    metrics = _normalize_metric_names(args.metrics.split(","))
    group_fields = [item.strip() for item in args.group_by.split(",") if item.strip()]

    run_dirs = discover_run_dirs(
        roots=roots,
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
        run_range=run_range,
    )
    if args.limit_runs > 0:
        run_dirs = run_dirs[: args.limit_runs]

    payloads = []
    invalid_runs = []
    for run_dir in run_dirs:
        payload = _load_run_artifacts(run_dir)
        payload["data_status"] = _validate_run_payload(payload)
        payloads.append(payload)
        if payload["data_status"] == "invalid":
            invalid_runs.append(
                {
                    "run_name": payload["run_name"],
                    "run_dir": payload["run_dir"],
                    "issues": payload["issues"],
                }
            )

    catalog_rows = [build_catalog_entry(payload) for payload in payloads]
    catalog_rows.sort(key=lambda item: (item.get("created_at_ts", 0.0), _natural_run_key(item["run_name"])))

    analyzable_rows = [
        row for row in catalog_rows
        if row.get("data_status") == "ok" or (args.include_partial and row.get("data_status") == "partial")
    ]
    selected_metrics = _metric_candidates(metrics, analyzable_rows)
    descriptive_stats = _descriptive_stats(analyzable_rows, selected_metrics)
    group_analysis = _group_analysis(analyzable_rows, group_fields, selected_metrics)
    trend_analysis = _trend_analysis(analyzable_rows, selected_metrics)
    corr_matrix, top_correlations = _correlation_analysis(analyzable_rows, selected_metrics)
    plots = generate_plots(analyzable_rows, selected_metrics, group_fields, output_dir)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "roots": roots,
        "output_dir": output_dir,
        "total_scanned_runs": len(run_dirs),
        "valid_runs": len(analyzable_rows),
        "invalid_runs": len(invalid_runs),
        "metrics": selected_metrics,
        "group_by": group_fields,
        "descriptive_stats": descriptive_stats,
        "group_analysis": group_analysis,
        "trend_analysis": trend_analysis,
        "correlation_matrix": corr_matrix,
        "top_correlations": top_correlations,
        "plots": plots,
    }

    catalog_csv = os.path.join(output_dir, "run_catalog.csv")
    invalid_json = os.path.join(output_dir, "invalid_runs.json")
    inventory_json = os.path.join(output_dir, "data_inventory.json")
    report_json = os.path.join(output_dir, "aggregate_report.json")
    report_md = os.path.join(output_dir, "aggregate_report.md")

    _write_catalog_csv(catalog_csv, catalog_rows)
    _json_dump(invalid_json, invalid_runs)
    _json_dump(
        inventory_json,
        [
            {
                "run_name": payload["run_name"],
                "run_dir": payload["run_dir"],
                "data_status": payload["data_status"],
                "issues": payload["issues"],
                "structured_inventory": payload["structured_inventory"],
            }
            for payload in payloads
        ],
    )
    report["catalog_csv"] = catalog_csv
    report["invalid_runs_json"] = invalid_json
    report["data_inventory_json"] = inventory_json
    _json_dump(report_json, report)
    write_markdown_report(report_md, report)

    print("=" * 72)
    print("Aggregate run analysis complete")
    print("=" * 72)
    print(f"Roots: {roots}")
    print(f"Scanned runs: {len(run_dirs)}")
    print(f"Analyzed runs: {len(analyzable_rows)}")
    print(f"Invalid runs: {len(invalid_runs)}")
    print(f"Catalog: {catalog_csv}")
    print(f"Report:  {report_json}")
    print(f"Markdown report: {report_md}")
    if plots:
        print("Plots:")
        for path in plots:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
