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


def parse_args():
    parser = argparse.ArgumentParser('Meta-AQNODE MAML')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--tasks-per-epoch', type=int, default=5)
    parser.add_argument('--outer-lr', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--inner-lr', type=float, default=0.1)
    parser.add_argument('--inner-steps', type=int, default=5)
    parser.add_argument('--lambda-meta-lr', type=float, default=0.2)
    parser.add_argument('--lambda-supervision-weight', type=float, default=1.0)
    parser.add_argument('--trend-weight', type=float, default=0.5)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--seq-len', type=int, default=100)

    parser.add_argument('--perturb-scale', type=float, default=0.05)
    parser.add_argument('--num-train-tasks', type=int, default=12)
    parser.add_argument('--num-test-tasks', type=int, default=5)
    parser.add_argument('--num-traj-per-task', type=int, default=200)
    parser.add_argument('--T', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--w-z', type=float, default=1.5)
    parser.add_argument('--w-gamma', type=float, default=1.2)
    parser.add_argument('--bloch-reg', type=float, default=0.01)
    parser.add_argument('--init-weight', type=float, default=0.1)
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of train tasks held out for validation')
    parser.add_argument('--save-dir', type=str, default='./meta_aqnode_results')
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

    print("=" * 60)
    print("Meta-AQNODE Phase 1: Training (MAML)")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Inner loop: {args.inner_steps} steps x lr={args.inner_lr}")
    print(f"Outer loop lr: {args.outer_lr}")
    print(f"Lambda meta lr: {args.lambda_meta_lr}")
    print(f"Lambda supervision weight: {args.lambda_supervision_weight}")
    print(f"Trend weight: {args.trend_weight}")
    print(f"Tasks: {args.num_train_tasks} train + {args.num_test_tasks} test")

    # ==============================
    # 1. Generate task datasets
    # ==============================
    print("\n" + "-" * 60)
    print("Generating task datasets...")

    train_params = []
    for _ in range(args.num_train_tasks):
        alpha = np.random.uniform(0.4, 0.7)
        r = np.random.uniform(0.2, 0.5)
        train_params.append((alpha, r))

    test_params = []
    for _ in range(args.num_test_tasks):
        alpha = np.random.uniform(0.2, 0.8)
        r = np.random.uniform(0.1, 0.6)
        test_params.append((alpha, r))

    print(f"  Train params: {len(train_params)} tasks (alpha in [0.4,0.7], r in [0.2,0.5])")
    print(f"  Test  params: {len(test_params)} tasks (alpha in [0.2,0.8], r in [0.1,0.6], OOD)")

    train_dataset = build_task_dataset(
        train_params, num_traj=args.num_traj_per_task,
        T=args.T, dt=args.dt, verbose=True)
    test_dataset = build_task_dataset(
        test_params, num_traj=max(50, args.num_traj_per_task // 4),
        T=args.T, dt=args.dt, verbose=True)

    # Hold out validation split
    train_ids = list(train_dataset.keys())
    np.random.shuffle(train_ids)
    n_val = max(1, int(len(train_ids) * args.val_ratio))
    val_ids = train_ids[:n_val]
    train_ids = train_ids[n_val:]
    val_tasks = {tid: train_dataset[tid] for tid in val_ids}
    train_tasks = {tid: train_dataset[tid] for tid in train_ids}

    print(f"\n  Train tasks: {len(train_tasks)}, Val tasks: {len(val_tasks)}")

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
        perturb_scale=args.perturb_scale,
        w_z=args.w_z, w_gamma=args.w_gamma,
        bloch_reg=args.bloch_reg,
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        lambda_meta_lr=args.lambda_meta_lr,
        lambda_supervision_weight=args.lambda_supervision_weight,
        trend_weight=args.trend_weight,
        init_weight=args.init_weight,
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
    print(f"{'Task':<6} {'alpha':<8} {'r':<8} {'Loss':<12} {'BlochMSE':<12}")
    print("-" * 47)
    for tid, res in test_results.items():
        print(
            f"{tid:<6} {res['alpha']:<8.2f} {res['r']:<8.2f} "
            f"{res['loss']:<12.6f} {res['bloch_mse']:<12.6f}"
        )

    avg_loss = np.mean([r['loss'] for r in test_results.values()])
    avg_bloch_mse = np.mean([r['bloch_mse'] for r in test_results.values()])
    print(f"\nAverage test loss: {avg_loss:.6f}")
    print(f"Average test Bloch MSE: {avg_bloch_mse:.6f}")

    # ==============================
    # 5. Plot
    # ==============================
    plot_results(trainer, test_dataset, test_results, args.save_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
