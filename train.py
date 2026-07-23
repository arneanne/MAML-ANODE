# train.py — Meta-AQNODE: MAML meta-learning training

import os
import sys
import torch
import numpy as np
import argparse
import json
import time


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iMODE'))
from analysis import generate_analysis_report
from experiment_logger import ExperimentLogger
from models import MetaTrainer
from visualization import plot_results


def build_arg_parser():
    default_train_data_dir = os.path.join(os.path.dirname(__file__), "exported_tasks")
    default_test_data_dir = os.path.join(os.path.dirname(__file__), "exported_test_tasks")
    parser = argparse.ArgumentParser('Meta-AQNODE MAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--tasks-per-epoch', type=int, default=5,
                        help='Number of train tasks randomly sampled per epoch; <=0 means use all train tasks')
    parser.add_argument('--outer-lr', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--inner-lr', type=float, default=1.0)
    parser.add_argument('--inner-steps', type=int, default=5)
    parser.add_argument('--context-dim', type=int, default=2)
    parser.add_argument('--eta-reg-weight', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--T', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--w-alpha', '--w-a', dest='w_alpha', type=float, default=1.2)
    parser.add_argument('--w-r', type=float, default=1.5)
    parser.add_argument('--w-alpha-frac', type=float, default=1.2,
                        help='Weight for alpha^2 / (1 + r^2) loss term')
    parser.add_argument('--w-r-frac', type=float, default=1.5,
                        help='Weight for r^2 / (1 + r^2) loss term')
    parser.add_argument('--train-ood-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of train tasks held out for validation')
    parser.add_argument('--val-param-weight', type=float, default=1.0,
                        help='Checkpoint selection weight for validation parameter MAE')
    parser.add_argument('--val-ood-mse-weight', type=float, default=1.0,
                        help='Checkpoint selection weight for validation worst-axis RMSE')
    parser.add_argument('--val-bloch-weight', type=float, default=0.25,
                        help='Checkpoint selection tie-breaker on validation Bloch RMSE')
    parser.add_argument('--train-data-dir', type=str, default=default_train_data_dir,
                        help='Directory containing exported training task .pt files')
    parser.add_argument('--test-data-dir', type=str, default=default_test_data_dir,
                        help='Directory containing exported test task .pt files')
    parser.add_argument('--run-root', type=str, default='./results')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    return parser


def parse_args():
    return build_arg_parser().parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_task_dataset_from_dir(data_dir, split_name):
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"{split_name.capitalize()} data directory not found: {data_dir}. "
            f"Run export_trajectory_dataset.py first or pass --{split_name}-data-dir."
        )

    file_entries = []
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        for item in manifest.get("tasks", []):
            file_name = item.get("file_name")
            file_path = item.get("file_path")
            if file_path and os.path.isfile(file_path):
                file_entries.append((file_name or os.path.basename(file_path), file_path))
            elif file_name:
                file_entries.append((file_name, os.path.join(data_dir, file_name)))

    if not file_entries:
        file_names = sorted(
            name for name in os.listdir(data_dir)
            if name.endswith(".pt") and name.startswith("task_alpha_")
        )
        file_entries = [(name, os.path.join(data_dir, name)) for name in file_names]

    if not file_entries:
        raise FileNotFoundError(
            f"No exported task .pt files found in {data_dir}."
        )

    dataset = {}
    train_params = []
    for idx, (_, file_path) in enumerate(file_entries):
        task_data = torch.load(file_path, map_location="cpu")
        dataset[idx] = task_data
        train_params.append((float(task_data["alpha"]), float(task_data["r"])))

    return dataset, train_params, manifest


def main():
    run_start_time = time.perf_counter()
    args = parse_args()
    default_args = vars(build_arg_parser().parse_args([]))
    seed_everything(args.seed)

    os.makedirs(args.run_root, exist_ok=True)
    if args.run_name is None:
        existing = [
            d for d in os.listdir(args.run_root)
            if os.path.isdir(os.path.join(args.run_root, d)) and d.startswith("run_")
        ]
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_", 1)[1]))
            except Exception:
                pass
        next_id = (max(nums) + 1) if nums else 1
        run_name = f"run_{next_id:03d}"
    else:
        run_name = args.run_name

    run_dir = os.path.join(args.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    if args.save_dir is None:
        save_dir = os.path.join(run_dir, "meta_aqnode_results")
    else:
        save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    experiment_logger = ExperimentLogger(run_dir)
    experiment_logger.log_config(
        args_dict=vars(args),
        default_dict=default_args,
        run_name=run_name,
        run_dir=run_dir,
        save_dir=save_dir,
    )

    print("=" * 60)
    print("Meta-AQNODE Phase 1: Training (physics-guided meta-parameter inference)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Inner loop: {args.inner_steps} steps x lr={args.inner_lr}")
    print(f"Outer loop lr: {args.outer_lr}")
    print(f"Context dim: {args.context_dim}")
    print(f"Eta regularisation weight: {args.eta_reg_weight}")
    print(
        "Loss: "
        f"w_alpha*alpha + w_r*r + "
        f"w_alpha_frac*alpha^2/(1+r^2) + w_r_frac*r^2/(1+r^2)"
    )
    print("Dynamics: eta -> alpha,r -> analytical Delta/gamma -> Bloch ODE")
    print("Task variable: eta = [raw_alpha, raw_r], initialised from support statistics")
    print(f"Run dir: {os.path.abspath(run_dir)}")
    print(f"Save dir: {os.path.abspath(save_dir)}")

    # ==============================
    # 1. Load task datasets
    # ==============================
    print("\n" + "-" * 60)
    print("Loading task datasets...")

    train_dataset, train_params, train_manifest = load_task_dataset_from_dir(args.train_data_dir, "train")
    actual_num_train_tasks = len(train_params)
    num_time_steps = len(np.arange(0, args.T, args.dt))
    random_window_enabled = args.seq_len < num_time_steps

    test_dataset, test_params, test_manifest = load_task_dataset_from_dir(args.test_data_dir, "test")
    actual_num_test_tasks = len(test_params)

    print(
        f"  Train data: loaded {actual_num_train_tasks} tasks from {os.path.abspath(args.train_data_dir)}"
    )
    for idx, (alpha, r) in enumerate(train_params):
        print(f"    T{idx}: alpha={alpha:.2f}, r={r:.2f}")
    print(
        f"  Test data: loaded {actual_num_test_tasks} tasks from {os.path.abspath(args.test_data_dir)}"
    )
    for idx, (alpha, r) in enumerate(test_params[:min(10, len(test_params))]):
        print(f"    E{idx}: alpha={alpha:.2f}, r={r:.2f}")
    if len(test_params) > 10:
        print(f"    ... {len(test_params) - 10} more test tasks")
    print(
        f"  Sequence setup: seq_len={args.seq_len}, time_steps={num_time_steps}, "
        f"random_windows={'enabled' if random_window_enabled else 'disabled'}"
    )

    with open(os.path.join(run_dir, "train_curriculum.json"), "w", encoding="utf-8") as f:
        json.dump(
            [
                {"task": idx, "alpha": float(alpha), "r": float(r)}
                for idx, (alpha, r) in enumerate(train_params)
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Hold out a reproducible random validation subset for checkpoint selection.
    all_train_ids = sorted(train_dataset.keys())
    requested_val = int(round(len(all_train_ids) * args.val_ratio)) if args.val_ratio > 0 else 0
    max_val = max(len(all_train_ids) - 1, 0)
    val_count = min(max(requested_val, 0), max_val)
    if val_count > 0:
        split_rng = np.random.default_rng(args.seed)
        shuffled_ids = split_rng.permutation(all_train_ids).tolist()
        val_ids = sorted(shuffled_ids[:val_count])
        train_ids = sorted(shuffled_ids[val_count:])
    else:
        val_ids = []
        train_ids = all_train_ids

    train_tasks = {tid: train_dataset[tid] for tid in train_ids}
    val_tasks = {tid: train_dataset[tid] for tid in val_ids} if val_ids else None
    effective_tasks_per_epoch = len(train_ids) if args.tasks_per_epoch <= 0 else min(args.tasks_per_epoch, len(train_ids))

    split_payload = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "val_ratio": args.val_ratio,
        "actual_num_train_tasks": actual_num_train_tasks,
        "train_data_dir": os.path.abspath(args.train_data_dir),
        "train_manifest_total_tasks": (
            int(train_manifest.get("total_tasks", actual_num_train_tasks))
            if train_manifest is not None else actual_num_train_tasks
        ),
        "actual_num_test_tasks": actual_num_test_tasks,
        "test_data_dir": os.path.abspath(args.test_data_dir),
        "test_manifest_total_tasks": (
            int(test_manifest.get("total_tasks", actual_num_test_tasks))
            if test_manifest is not None else actual_num_test_tasks
        ),
        "tasks_per_epoch_requested": args.tasks_per_epoch,
        "tasks_per_epoch_effective": effective_tasks_per_epoch,
        "num_time_steps": num_time_steps,
        "random_window_enabled": random_window_enabled,
        "w_alpha": args.w_alpha,
        "w_r": args.w_r,
        "w_alpha_frac": args.w_alpha_frac,
        "w_r_frac": args.w_r_frac,
        "val_split_strategy": "seeded_random",
        "val_split_seed": args.seed,
        "run_name": run_name,
        "save_dir": os.path.abspath(save_dir),
        "selection_note": "validation uses a seeded random held-out subset of loaded train tasks for checkpoint selection",
    }
    with open(os.path.join(run_dir, "task_split.json"), "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)
    experiment_logger.log_dataset_summary(
        train_params=train_params,
        test_params=test_params,
        train_ids=train_ids,
        val_ids=val_ids,
        tasks_per_epoch_effective=effective_tasks_per_epoch,
        num_time_steps=num_time_steps,
        random_window_enabled=random_window_enabled,
    )

    print(f"\n  Train tasks: {len(train_tasks)}, Val tasks: {len(val_ids)}")
    print(f"  Tasks per epoch: requested={args.tasks_per_epoch}, effective={effective_tasks_per_epoch}")
    if val_ids:
        print(f"  Val split: seeded random task ids {val_ids} (seed={args.seed})")
    else:
        print("  Val split: disabled")

    # ==============================
    # 2. Initialise MetaTrainer
    # ==============================
    print("\n" + "-" * 60)
    print("Initialising MetaTrainer...")

    trainer = MetaTrainer(
        task_dataset=train_tasks,
        latent_dim=args.latent_dim,
        outer_lr=args.outer_lr,
        weight_decay=args.weight_decay,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        w_alpha=args.w_alpha, w_r=args.w_r,
        w_alpha_frac=args.w_alpha_frac,
        w_r_frac=args.w_r_frac,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        context_dim=args.context_dim,
        eta_reg_weight=args.eta_reg_weight,
        val_param_weight=args.val_param_weight,
        val_ood_mse_weight=args.val_ood_mse_weight,
        val_bloch_weight=args.val_bloch_weight,
        device=args.device)

    param_count = sum(p.numel() for p in trainer.meta_net.parameters())
    print(f"  InnerAQNode: {param_count:,} params")
    print(f"  Device: {trainer.device}")
    print(f"  Using actual device: {trainer.device}")

    # ==============================
    # 3. Train
    # ==============================
    print("\n" + "-" * 60)
    print("Meta-training...")

    train_start_time = time.perf_counter()
    trainer.train(num_epochs=args.num_epochs,
                  tasks_per_epoch=args.tasks_per_epoch,
                  val_tasks=val_tasks)
    train_time_sec = float(time.perf_counter() - train_start_time)
    experiment_logger.log_training_summary(trainer)

    # ==============================
    # 4. Test OOD
    # ==============================
    print("\n" + "-" * 60)
    print("Testing on OOD tasks...")

    test_start_time = time.perf_counter()
    test_results = trainer.eval_on_all_tasks(test_dataset)
    test_time_sec = float(time.perf_counter() - test_start_time)

    print(f"\nTest Results Summary:")
    print(
        f"{'Task':<6} {'A':<8} {'alpha':<8} {'r':<8} {'Loss':<12} {'BlochMSE':<12} "
        f"{'Mx':<10} {'My':<10} {'Mz':<10} {'|eta|':<10}"
    )
    print("-" * 98)
    for tid, res in test_results.items():
        mse_x = res.get('mse_x', float('nan'))
        mse_y = res.get('mse_y', float('nan'))
        mse_z = res.get('mse_z', float('nan'))
        eta = np.asarray(res.get('eta', []), dtype=float)
        eta_norm = float(np.linalg.norm(eta)) if eta.size > 0 else float('nan')
        print(
            f"{tid:<6} {res['A']:<8.3f} {res['alpha']:<8.2f} {res['r']:<8.2f} "
            f"{res['loss']:<12.6f} {res['bloch_mse']:<12.6f} "
            f"{mse_x:<10.6f} {mse_y:<10.6f} {mse_z:<10.6f} "
            f"{eta_norm:<10.6f}"
        )

    avg_loss = np.mean([r['loss'] for r in test_results.values()])
    avg_bloch_mse = np.mean([r['bloch_mse'] for r in test_results.values()])
    print(f"\nAverage test loss: {avg_loss:.6f}")
    print(f"Average test Bloch MSE: {avg_bloch_mse:.6f}")

    max_mx = np.nanmax([v.get("mse_x", np.nan) for v in test_results.values()])
    max_my = np.nanmax([v.get("mse_y", np.nan) for v in test_results.values()])
    max_mz = np.nanmax([v.get("mse_z", np.nan) for v in test_results.values()])
    max_all = np.nanmax([max_mx, max_my, max_mz])
    print(
        f"Max per-task MSE: x={max_mx:.6g}, y={max_my:.6g}, z={max_mz:.6g}"
    )
    print(f"Target check (all tasks < 1e-4): {'PASS' if max_all < 1e-4 else 'FAIL'}")
    experiment_logger.log_test_summary(test_results)

    def _jsonify(obj):
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if torch.is_tensor(obj):
            return obj.detach().cpu().tolist()
        if isinstance(obj, dict):
            return {str(k): _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonify(v) for v in obj]
        return str(obj)

    effective_config = {
        "run_name": run_name,
        "run_dir": os.path.abspath(run_dir),
        "save_dir": os.path.abspath(save_dir),
        "actual_num_train_tasks": actual_num_train_tasks,
        "train_data_dir": os.path.abspath(args.train_data_dir),
        "actual_num_test_tasks": actual_num_test_tasks,
        "test_data_dir": os.path.abspath(args.test_data_dir),
        "tasks_per_epoch_requested": args.tasks_per_epoch,
        "tasks_per_epoch_effective": effective_tasks_per_epoch,
        "num_time_steps": num_time_steps,
        "random_window_enabled": random_window_enabled,
        "val_count": len(val_ids),
        "train_count": len(train_ids),
    }
    config_payload = {
        "user_config": _jsonify(vars(args)),
        "effective_config": _jsonify(effective_config),
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)
    test_results_path = os.path.join(run_dir, "test_results.json")
    with open(test_results_path, "w", encoding="utf-8") as f:
        json.dump(_jsonify(test_results), f, ensure_ascii=False, indent=2)
    attention_weights_path = os.path.join(run_dir, "attention_weights.json")
    attention_payload = {
        str(tid): {
            "alpha": _jsonify(res.get("alpha")),
            "r": _jsonify(res.get("r")),
            "support_traj_idx": _jsonify(res.get("support_traj_idx", [])),
            "attention_weights": _jsonify(res.get("attention_weights", [])),
        }
        for tid, res in test_results.items()
    }
    with open(attention_weights_path, "w", encoding="utf-8") as f:
        json.dump(attention_payload, f, ensure_ascii=False, indent=2)
    embedding_diagnostics_path = os.path.join(run_dir, "embedding_diagnostics.json")
    embedding_payload = {
        str(tid): {
            "alpha": _jsonify(res.get("alpha")),
            "r": _jsonify(res.get("r")),
            "loss": _jsonify(res.get("loss")),
            "support_traj_idx": _jsonify(res.get("support_traj_idx", [])),
            "traj_embed_pairwise_cos_mean": _jsonify(res.get("traj_embed_pairwise_cos_mean")),
            "traj_embed_norm_mean": _jsonify(res.get("traj_embed_norm_mean")),
            "traj_embed_norm_std": _jsonify(res.get("traj_embed_norm_std")),
            "attention_logit_std": _jsonify(res.get("attention_logit_std")),
        }
        for tid, res in test_results.items()
    }
    with open(embedding_diagnostics_path, "w", encoding="utf-8") as f:
        json.dump(embedding_payload, f, ensure_ascii=False, indent=2)
    summary_path = os.path.join(run_dir, "summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("task,alpha,r,A,loss,bloch_mse,mse_x,mse_y,mse_z,latent_norm,eta\n")
        for tid, res in test_results.items():
            f.write(
                f"{tid},{res['alpha']},{res['r']},"
                f"{res['A']},{res['loss']},{res['bloch_mse']},"
                f"{res.get('mse_x','')},{res.get('mse_y','')},{res.get('mse_z','')},"
                f"{res.get('latent_norm','')},\"{res.get('eta','')}\"\n"
            )
    analysis_outputs = generate_analysis_report(run_dir, trainer, test_results)

    # ==============================
    # 5. Plot
    # ==============================
    plot_results(trainer, test_dataset, test_results, save_dir)
    experiment_logger.log_artifacts([
        os.path.join(run_dir, "train_curriculum.json"),
        os.path.join(run_dir, "task_split.json"),
        config_path,
        test_results_path,
        attention_weights_path,
        embedding_diagnostics_path,
        summary_path,
        analysis_outputs.get("report_json"),
        analysis_outputs.get("report_md"),
        analysis_outputs.get("training_history_json"),
        analysis_outputs.get("training_history_csv"),
        os.path.join(run_dir, "analysis", "alpha_r_prediction_map.png"),
        os.path.join(save_dir, "attention_diagnostics.png"),
        os.path.join(run_dir, "experiment_log.jsonl"),
        os.path.join(run_dir, "experiment_log.md"),
        os.path.join(save_dir, "training_loss.png"),
    ])
    run_total_time_sec = float(time.perf_counter() - run_start_time)
    per_epoch_times_sec = list(getattr(trainer, "epoch_time_his", []))
    avg_epoch_time_sec = float(np.mean(per_epoch_times_sec)) if per_epoch_times_sec else None
    print("\nTiming Summary:")
    print(f"  Run total time: {run_total_time_sec:.2f}s")
    print(f"  Train time: {train_time_sec:.2f}s")
    print(f"  Test time: {test_time_sec:.2f}s")
    if avg_epoch_time_sec is not None:
        print(f"  Per-epoch time: avg={avg_epoch_time_sec:.2f}s")
    else:
        print("  Per-epoch time: n/a")
    experiment_logger.log_timing_summary(
        run_total_sec=run_total_time_sec,
        train_time_sec=train_time_sec,
        test_time_sec=test_time_sec,
        per_epoch_times_sec=per_epoch_times_sec,
    )
    print(f"Analysis report: {analysis_outputs['report_md']}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
