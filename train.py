# train.py — Meta-AQNODE: MAML meta-learning training

import os
import sys
import torch
import numpy as np
import argparse



sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'iMODE'))
from data_gen import generate_task_data, build_task_dataset
from models import MetaTrainer
from visualization import plot_results


def get_fixed_train_curriculum(num_tasks=50):
    """Deterministic fixed training tasks ordered from hard to easy.

    Hardness heuristic:
      - farther from the central training region is harder
      - boundary/extreme alpha or r is harder
    """
    alpha_grid = np.linspace(0.22, 0.78, 10)
    r_grid = np.linspace(0.12, 0.58, 5)
    center_alpha = 0.52
    center_r = 0.35

    candidates = []
    for alpha in alpha_grid:
        for r in r_grid:
            # Normalize distance by parameter ranges to get a balanced hardness score.
            alpha_dist = abs(alpha - center_alpha) / (0.78 - 0.22)
            r_dist = abs(r - center_r) / (0.58 - 0.12)
            hardness = alpha_dist + r_dist
            candidates.append((hardness, (float(alpha), float(r))))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [params for _, params in candidates[:num_tasks]]


def parse_args():
    parser = argparse.ArgumentParser('Meta-AQNODE MAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--tasks-per-epoch', type=int, default=50,
                        help='Deprecated compatibility flag; each epoch now sweeps all train tasks in id order')
    parser.add_argument('--outer-lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--inner-lr', type=float, default=0.05)
    parser.add_argument('--inner-steps', type=int, default=5)
    parser.add_argument('--context-dim', type=int, default=2)
    parser.add_argument('--eta-reg-weight', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=100)
    parser.add_argument('--num-train-tasks', type=int, default=50)
    parser.add_argument('--num-test-tasks', type=int, default=100)
    parser.add_argument('--num-traj-per-task', type=int, default=80)
    parser.add_argument('--T', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--w-x', type=float, default=0.5)
    parser.add_argument('--w-y', type=float, default=0.5)
    parser.add_argument('--w-z', type=float, default=0.5)
    parser.add_argument('--w-alpha', '--w-a', dest='w_alpha', type=float, default=1.2)
    parser.add_argument('--w-r', type=float, default=1.2)
    parser.add_argument('--w-dy', type=float, default=0.2)
    parser.add_argument('--train-ood-ratio', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.1,
                        help='Fraction of train tasks held out for validation')
    parser.add_argument('--run-root', type=str, default='./results')
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    return parser.parse_args()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
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
        args.run_name = f"run_{next_id:03d}"

    run_dir = os.path.join(args.run_root, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    if args.save_dir is None:
        args.save_dir = os.path.join(run_dir, "meta_aqnode_results")
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("Meta-AQNODE Phase 1: Training (physics-guided meta-parameter inference)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Inner loop: {args.inner_steps} steps x lr={args.inner_lr}")
    print(f"Outer loop lr: {args.outer_lr}")
    print(f"Context dim: {args.context_dim}")
    print(f"Eta regularisation weight: {args.eta_reg_weight}")
    print("Loss: weighted MSE on [x, y, z, alpha, r], with eta regularisation")
    print("Dynamics: eta -> alpha,r -> analytical Delta/gamma -> Bloch ODE")
    print("Task variable: eta = [raw_alpha, raw_r], initialised from support statistics")
    print(f"Tasks: {args.num_train_tasks} train + {args.num_test_tasks} test")
    print(f"Run dir: {os.path.abspath(run_dir)}")
    print(f"Save dir: {os.path.abspath(args.save_dir)}")

    # ==============================
    # 1. Generate task datasets
    # ==============================
    print("\n" + "-" * 60)
    print("Generating task datasets...")

    train_params = get_fixed_train_curriculum(args.num_train_tasks)
    args.num_train_tasks = len(train_params)

    test_params = []
    for _ in range(args.num_test_tasks):
        alpha = np.random.uniform(0.2, 0.8)
        r = np.random.uniform(0.1, 0.6)
        test_params.append((alpha, r))

    print(f"  Train params: {len(train_params)} fixed tasks (hard -> easy)")
    for idx, (alpha, r) in enumerate(train_params):
        print(f"    T{idx}: alpha={alpha:.2f}, r={r:.2f}")
    print(f"  Test  params: {len(test_params)} tasks (alpha in [0.2,0.8], r in [0.1,0.6], OOD)")

    import json
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

    train_dataset = build_task_dataset(
        train_params, num_traj=args.num_traj_per_task,
        T=args.T, dt=args.dt, verbose=True)
    test_dataset = build_task_dataset(
        test_params, num_traj=max(50, args.num_traj_per_task // 4),
        T=args.T, dt=args.dt, verbose=True)

    # Keep all fixed curriculum tasks for training in hard->easy order.
    train_ids = sorted(train_dataset.keys())
    train_tasks = {tid: train_dataset[tid] for tid in train_ids}
    val_tasks = None

    print(f"\n  Train tasks: {len(train_tasks)}, Val tasks: 0")

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
        w_x=args.w_x, w_y=args.w_y, w_z=args.w_z,
        w_alpha=args.w_alpha, w_r=args.w_r,
        w_dy=args.w_dy,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        context_dim=args.context_dim,
        eta_reg_weight=args.eta_reg_weight,
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

    trainer.train(num_epochs=args.num_epochs,
                  tasks_per_epoch=args.tasks_per_epoch,
                  val_tasks=val_tasks)

    # ==============================
    # 4. Test OOD
    # ==============================
    print("\n" + "-" * 60)
    print("Testing on OOD tasks...")

    test_results = trainer.eval_on_all_tasks(test_dataset)

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

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(vars(args)), f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "test_results.json"), "w", encoding="utf-8") as f:
        json.dump(_jsonify(test_results), f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "summary.csv"), "w", encoding="utf-8") as f:
        f.write("task,alpha,r,A,loss,bloch_mse,mse_x,mse_y,mse_z,latent_norm,eta\n")
        for tid, res in test_results.items():
            f.write(
                f"{tid},{res['alpha']},{res['r']},"
                f"{res['A']},{res['loss']},{res['bloch_mse']},"
                f"{res.get('mse_x','')},{res.get('mse_y','')},{res.get('mse_z','')},"
                f"{res.get('latent_norm','')},\"{res.get('eta','')}\"\n"
            )

    # ==============================
    # 5. Plot
    # ==============================
    plot_results(trainer, test_dataset, test_results, args.save_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
