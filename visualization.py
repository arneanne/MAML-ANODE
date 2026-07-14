# visualization.py - Bloch sphere and training visualization
import json
import os, numpy as np, matplotlib, torch, matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_bloch_sphere(ax, true_traj=None, pred_traj=None, title=''):
    r = 1.0

    # Sphere surface
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
    ax.plot_surface(r*np.sin(v)*np.cos(u), r*np.sin(v)*np.sin(u), r*np.cos(v),
                    color='lightgray', alpha=0.06, rstride=2, cstride=2, lw=0)
    ax.plot_wireframe(r*np.sin(v)*np.cos(u), r*np.sin(v)*np.sin(u), r*np.cos(v),
                      color='gray', alpha=0.25, rstride=4, cstride=4, lw=0.4)

    # Equator
    u_eq = np.linspace(0, 2*np.pi, 80)
    ax.plot(np.cos(u_eq), np.sin(u_eq), np.zeros_like(u_eq),
            color='gray', lw=0.8, alpha=0.6)

    # Latitude rings
    for th in np.radians([30, 60, 120, 150]):
        u_ring = np.linspace(0, 2*np.pi, 60)
        ax.plot(r*np.sin(th)*np.cos(u_ring), r*np.sin(th)*np.sin(u_ring),
                r*np.cos(th)*np.ones_like(u_ring), color='gray', lw=0.5, alpha=0.4)

    # Longitude meridians
    for ph in np.radians([0, 45, 90, 135, 180, 225, 270, 315]):
        v_mer = np.linspace(0, np.pi, 40)
        ax.plot(r*np.sin(v_mer)*np.cos(ph), r*np.sin(v_mer)*np.sin(ph),
                r*np.cos(v_mer), color='gray', lw=0.4, alpha=0.3)

    # Axis lines
    for (xs, ys, zs), lbl in [(([-1.15,1.15],[0,0],[0,0]),'x'),
                              (([0,0],[-1.15,1.15],[0,0]),'y'),
                              (([0,0],[0,0],[-1.15,1.15]),'z')]:
        ax.plot(xs, ys, zs, color='gray', lw=1.2)
        ax.text(xs[1], ys[1], zs[1] + 0.05, lbl, fontsize=12, ha='center', va='center')
    ax.text(0, 0, 1.25, r'$|0\rangle$', fontsize=14, ha='center', va='center')
    ax.text(0, 0, -1.25, r'$|1\rangle$', fontsize=14, ha='center', va='center')

    if true_traj is not None and len(true_traj) > 1:
        t = true_traj
        ax.plot(t[:,0], t[:,1], t[:,2], color='blue', lw=3, alpha=0.9)
        ax.scatter(*t[0], c='blue', s=40, marker='o', zorder=5)
        ax.scatter(*t[-1], c='blue', s=60, marker='s', zorder=5)
    if pred_traj is not None and len(pred_traj) > 1:
        p = pred_traj
        ax.plot(p[:,0], p[:,1], p[:,2], color='red', lw=3, alpha=0.9,
                linestyle='dashed', dashes=[5, 3])
        ax.scatter(*p[0], c='red', s=40, marker='o', zorder=5)
        ax.scatter(*p[-1], c='red', s=60, marker='s', zorder=5)

    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_zlim(-1.1, 1.1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=30, azim=-55)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=10)

    handles = [
        mlines.Line2D([0], [0], color='blue', lw=3, label='Real'),
        mlines.Line2D([0], [0], color='red', lw=3, label='Predicted',
                      linestyle='dashed'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=10)


def plot_component_panels(
    t,
    true_traj,
    pred_traj,
    true_delta,
    pred_delta,
    true_gamma,
    pred_gamma,
    title,
    save_path,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()
    labels = ['x', 'y', 'z']

    for idx, label in enumerate(labels):
        ax = axes[idx]
        ax.plot(t, true_traj[:, idx], color='blue', lw=2.2, label='Real')
        ax.plot(
            t,
            pred_traj[:, idx],
            color='red',
            lw=2.0,
            linestyle='dashed',
            dashes=[5, 3],
            label='Predicted',
        )
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_title(title, fontsize=10)
            ax.legend(loc='best', fontsize=9)

    aux_ax = axes[3]
    aux_ax.plot(t, true_delta, color='green', lw=2.0, label='Real delta')
    aux_ax.plot(
        t,
        pred_delta,
        color='limegreen',
        lw=1.9,
        linestyle='dashed',
        dashes=[5, 3],
        label='Pred delta',
    )
    aux_ax.plot(t, true_gamma, color='purple', lw=2.0, label='Real gamma')
    aux_ax.plot(
        t,
        pred_gamma,
        color='magenta',
        lw=1.9,
        linestyle='dashed',
        dashes=[5, 3],
        label='Pred gamma',
    )
    aux_ax.set_ylabel('delta/gamma')
    aux_ax.grid(True, alpha=0.3)
    aux_ax.legend(loc='best', fontsize=8)

    axes[2].set_xlabel('t')
    axes[3].set_xlabel('t')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_training_diagnostics(trainer, save_dir):
    epochs = np.arange(len(trainer.metrics_his))
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()

    axes[0].plot(epochs, trainer.metrics_his, lw=2.0, label='Train total')
    if trainer.val_metrics_his:
        axes[0].plot(epochs[:len(trainer.val_metrics_his)], trainer.val_metrics_his, lw=2.0, label='Val total')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(epochs[:len(trainer.train_recon_his)], trainer.train_recon_his, lw=2.0, label='Train recon')
    axes[1].plot(epochs[:len(trainer.train_param_his)], trainer.train_param_his, lw=2.0, label='Train param')
    if trainer.val_recon_his:
        axes[1].plot(epochs[:len(trainer.val_recon_his)], trainer.val_recon_his, '--', lw=1.8, label='Val recon')
    if trainer.val_param_his:
        axes[1].plot(epochs[:len(trainer.val_param_his)], trainer.val_param_his, '--', lw=1.8, label='Val param')
    axes[1].set_title('Recon vs Param')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    axes[2].plot(epochs[:len(trainer.train_eta_norm_his)], trainer.train_eta_norm_his, lw=2.0, color='tab:green')
    axes[2].set_title('Mean |eta|')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel('Epoch')

    axes[3].plot(epochs[:len(trainer.train_eta_shift_his)], trainer.train_eta_shift_his, lw=2.0, color='tab:red')
    axes[3].set_title('Mean eta shift')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel('Epoch')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_diagnostics.png'), dpi=150)
    plt.close()


def plot_parameter_diagnostics(test_results, save_dir):
    tids = sorted(test_results.keys())
    true_alpha = np.array([test_results[tid]['alpha'] for tid in tids], dtype=float)
    pred_alpha = np.array([test_results[tid]['pred_alpha'] for tid in tids], dtype=float)
    true_r = np.array([test_results[tid]['r'] for tid in tids], dtype=float)
    pred_r = np.array([test_results[tid]['pred_r'] for tid in tids], dtype=float)
    err_alpha = np.array([test_results[tid]['err_alpha'] for tid in tids], dtype=float)
    err_r = np.array([test_results[tid]['err_r'] for tid in tids], dtype=float)
    bloch_mse = np.array([test_results[tid]['bloch_mse'] for tid in tids], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    min_alpha = min(true_alpha.min(), pred_alpha.min())
    max_alpha = max(true_alpha.max(), pred_alpha.max())
    axes[0].scatter(true_alpha, pred_alpha, color='tab:blue')
    axes[0].plot([min_alpha, max_alpha], [min_alpha, max_alpha], 'k--', lw=1)
    axes[0].set_xlabel('True alpha')
    axes[0].set_ylabel('Pred alpha')
    axes[0].set_title('Alpha Calibration')
    axes[0].grid(True, alpha=0.3)

    min_r = min(true_r.min(), pred_r.min())
    max_r = max(true_r.max(), pred_r.max())
    axes[1].scatter(true_r, pred_r, color='tab:orange')
    axes[1].plot([min_r, max_r], [min_r, max_r], 'k--', lw=1)
    axes[1].set_xlabel('True r')
    axes[1].set_ylabel('Pred r')
    axes[1].set_title('r Calibration')
    axes[1].grid(True, alpha=0.3)

    axes[2].scatter(err_alpha, bloch_mse, color='tab:purple')
    axes[2].set_xlabel('|alpha error|')
    axes[2].set_ylabel('Bloch MSE')
    axes[2].set_title('Alpha Error vs Bloch MSE')
    axes[2].grid(True, alpha=0.3)

    axes[3].scatter(err_r, bloch_mse, color='tab:green')
    axes[3].set_xlabel('|r error|')
    axes[3].set_ylabel('Bloch MSE')
    axes[3].set_title('r Error vs Bloch MSE')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'parameter_diagnostics.png'), dpi=150)
    plt.close()


def plot_task_ranking(test_results, save_dir):
    tids = sorted(test_results.keys())
    labels = [f'T{tid}' for tid in tids]
    bloch = np.array([test_results[tid]['bloch_mse'] for tid in tids], dtype=float)
    err_alpha = np.array([test_results[tid]['err_alpha'] for tid in tids], dtype=float)
    err_r = np.array([test_results[tid]['err_r'] for tid in tids], dtype=float)
    mse_z = np.array([test_results[tid]['mse_z'] for tid in tids], dtype=float)

    order = np.argsort(-bloch)
    labels = [labels[idx] for idx in order]
    bloch = bloch[order]
    err_alpha = err_alpha[order]
    err_r = err_r[order]
    mse_z = mse_z[order]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = np.arange(len(labels))

    axes[0].bar(x, bloch, color='tab:red', alpha=0.8)
    axes[0].set_ylabel('Bloch MSE')
    axes[0].set_title('Task ranking by Bloch MSE')
    axes[0].grid(True, axis='y', alpha=0.3)

    width = 0.25
    axes[1].bar(x - width, err_alpha, width=width, label='|alpha error|', color='tab:blue')
    axes[1].bar(x, err_r, width=width, label='|r error|', color='tab:orange')
    axes[1].bar(x + width, mse_z, width=width, label='z MSE', color='tab:green')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Per-task parameter and z errors')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)
    axes[1].grid(True, axis='y', alpha=0.3)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'task_ranking.png'), dpi=150)
    plt.close()


def write_analysis_summary(test_results, save_dir):
    tids = sorted(test_results.keys())
    true_alpha = np.array([test_results[tid]['alpha'] for tid in tids], dtype=float)
    pred_alpha = np.array([test_results[tid]['pred_alpha'] for tid in tids], dtype=float)
    true_r = np.array([test_results[tid]['r'] for tid in tids], dtype=float)
    pred_r = np.array([test_results[tid]['pred_r'] for tid in tids], dtype=float)
    bloch = np.array([test_results[tid]['bloch_mse'] for tid in tids], dtype=float)
    mse_z = np.array([test_results[tid]['mse_z'] for tid in tids], dtype=float)
    err_alpha = np.array([test_results[tid]['err_alpha'] for tid in tids], dtype=float)
    err_r = np.array([test_results[tid]['err_r'] for tid in tids], dtype=float)
    eta_norm = np.array([
        float(np.linalg.norm(np.asarray(test_results[tid].get('eta', []), dtype=float)))
        for tid in tids
    ], dtype=float)

    worst_idx = int(np.argmax(bloch))
    best_idx = int(np.argmin(bloch))
    summary = {
        "avg_bloch_mse": float(np.mean(bloch)),
        "avg_mse_z": float(np.mean(mse_z)),
        "avg_err_alpha": float(np.mean(err_alpha)),
        "avg_err_r": float(np.mean(err_r)),
        "true_alpha_range": [float(true_alpha.min()), float(true_alpha.max())],
        "pred_alpha_range": [float(pred_alpha.min()), float(pred_alpha.max())],
        "true_r_range": [float(true_r.min()), float(true_r.max())],
        "pred_r_range": [float(pred_r.min()), float(pred_r.max())],
        "eta_norm_range": [float(eta_norm.min()), float(eta_norm.max())],
        "best_task_by_bloch_mse": {
            "task": int(tids[best_idx]),
            "bloch_mse": float(bloch[best_idx]),
        },
        "worst_task_by_bloch_mse": {
            "task": int(tids[worst_idx]),
            "bloch_mse": float(bloch[worst_idx]),
        },
    }
    with open(os.path.join(save_dir, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def plot_results(trainer, test_dataset, test_results, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Training loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(trainer.metrics_his, linewidth=2.0, label='Train')
    if trainer.val_metrics_his:
        ax.semilogy(trainer.val_metrics_his, linewidth=2.0,
                     label='Val', alpha=0.8)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Avg Loss')
    ax.set_title('Meta-Training Loss'); ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_loss.png'), dpi=150)
    plt.close()
    plot_training_diagnostics(trainer, save_dir)
    plot_parameter_diagnostics(test_results, save_dir)
    plot_task_ranking(test_results, save_dir)
    write_analysis_summary(test_results, save_dir)

    # Bloch sphere for each test task
    for tid in test_dataset:
        td = test_dataset[tid]
        full_metrics = trainer.predict_full_trajectory(td, traj_idx=0)
        t = td['t'].detach().cpu().numpy()
        true_traj = full_metrics['true_traj']
        pred_traj = full_metrics['pred_traj']
        true_delta = full_metrics['true_delta']
        pred_delta = full_metrics['pred_delta']
        true_gamma = full_metrics['true_gamma']
        pred_gamma = full_metrics['pred_gamma']
        loss = full_metrics['loss']
        bloch_mse = full_metrics['bloch_mse']
        mse_x = full_metrics.get('mse_x', 0.0)
        mse_y = full_metrics.get('mse_y', 0.0)
        mse_z = full_metrics.get('mse_z', 0.0)
        mse_delta = full_metrics.get('mse_delta', 0.0)
        mse_gamma = full_metrics.get('mse_gamma', 0.0)
        eta = full_metrics.get('eta')
        eta_norm = float(np.linalg.norm(eta)) if eta is not None else 0.0
        task_A = td["alpha"] ** 2 * td["r"] ** 2 / (1.0 + td["r"] ** 2)
        pred_A = full_metrics.get('pred_A', float('nan'))
        pred_alpha = full_metrics.get('pred_alpha', float('nan'))
        pred_r = full_metrics.get('pred_r', float('nan'))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ttl = (
            f'Task {tid}: alpha={td["alpha"]:.2f}->{pred_alpha:.2f}, '
            f'r={td["r"]:.2f}->{pred_r:.2f}, '
            f'A(diag)={task_A:.3f}->{pred_A:.3f}, '
            f'total={loss:.4f}, bloch={bloch_mse:.4f}, '
            f'x/y/z={mse_x:.3f}/{mse_y:.3f}/{mse_z:.3f}, '
            f'd/g(diag)={mse_delta:.3f}/{mse_gamma:.3f}, |eta|={eta_norm:.2f}'
        )
        draw_bloch_sphere(ax, true_traj, pred_traj, title=ttl)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bloch_task_{tid}.png'), dpi=150)
        plt.close()

        plot_component_panels(
            t=t,
            true_traj=true_traj,
            pred_traj=pred_traj,
            true_delta=true_delta,
            pred_delta=pred_delta,
            true_gamma=true_gamma,
            pred_gamma=pred_gamma,
            title=f'Task {tid} components (delta/gamma are formula-derived diagnostics)',
            save_path=os.path.join(save_dir, f'xyz_task_{tid}.png'),
        )

    print(f'\nResults saved to {save_dir}/')


# train.py — Meta-AQNODE: MAML meta-learning training

import os
import sys
import torch
import numpy as np
import argparse
