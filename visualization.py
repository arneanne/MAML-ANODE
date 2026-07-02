# visualization.py - Bloch sphere and training visualization
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
        lam = full_metrics['lam']

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ttl = (
            f'Task {tid}: alpha={td["alpha"]:.2f}, r={td["r"]:.2f}, '
            f'total={loss:.4f}, bloch={bloch_mse:.4f}, '
            f'lam=[{lam[0]:.2f}, {lam[1]:.2f}]'
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
            title=f'Task {tid} components',
            save_path=os.path.join(save_dir, f'xyz_task_{tid}.png'),
        )

    print(f'\nResults saved to {save_dir}/')


# train.py — Meta-AQNODE: MAML meta-learning training

import os
import sys
import torch
import numpy as np
import argparse
