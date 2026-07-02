# data_gen.py - synthetic quantum trajectory data for Meta-AQNODE
import os, tempfile, torch, numpy as np, random
from scipy.integrate import solve_ivp
from torchdiffeq import odeint

import os
import tempfile
import torch
import numpy as np
from scipy.integrate import solve_ivp
import random

def compute_delta_gamma(t, alpha, r, omega0=1.0, kBT=10.0):
    """Compute Delta(t) and gamma(t) using the legacy QNODE formulas.

    Args:
        t: time tensor, shape [N]
        alpha: system-environment coupling
        r: cutoff ratio omega_c/omega0
        omega0: qubit energy (fixed at 1.0 in Phase 1)
        kBT: kept for call compatibility; unused in the QNODE-aligned form

    Returns:
        Delta, gamma: tensors matching t shape
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float64)
    r2 = r ** 2

    # QNODE.py gamma(t) definition
    exp_term = torch.exp(-r * omega0 * t)
    cos_term = torch.cos(omega0 * t)
    sin_term = torch.sin(omega0 * t)
    gamma = (alpha ** 2 * omega0 * r2 / (1 + r2)) * (
        1 - exp_term * cos_term - r * sin_term
    )

    # QNODE.py Delta(t) definition
    Delta = (2 * alpha ** 2 * r2 / (1 + r2)) * (
        1 - exp_term * (cos_term - sin_term / r)
    )

    return Delta.float(), gamma.float()


def bloch_rhs(t, y, alpha, r, omega0, M, kBT):
    """RHS of Bloch equations (Eq.5).

    Args:
        t: current time (scalar)
        y: [x, y, z] Bloch vector

    Returns:
        [dx/dt, dy/dt, dz/dt]
    """
    x, y_, z = y
    t_tensor = torch.tensor([t], dtype=torch.float64)
    Delta, gamma = compute_delta_gamma(t_tensor, alpha, r, omega0, kBT)
    D = Delta.item()
    G = gamma.item()

    dx = -(D + M / 2.0) * x - omega0 * y_
    dy = omega0 * x - (D + M / 2.0) * y_
    dz = -2.0 * G - 2.0 * D * z

    return [dx, dy, dz]


def weak_measurement(y, M=0.4, zeta=0.9):
    """Weak measurement signal (Eq.7).

    dY/dt = sqrt(M * zeta) * z

    Returns:
        dY/dt scalar
    """
    return np.sqrt(M * zeta) * y[2]


def generate_task_data(alpha, r, num_traj=200, T=10.0, dt=0.01,
                       omega0=1.0, M=0.4, zeta=0.9, kBT=10.0, seed=None):
    """Generate multiple trajectories for one (alpha, r) task.

    Each trajectory:
      - Random initial Bloch vector
      - Integrate Bloch Eq.5 with RK45
      - Compute weak measurement Eq.7
      - Pre-compute analytical Delta(t), gamma(t) (Eq.2-3)

    Returns:
        dict with keys: bloch, delta, gamma, dY, t, alpha, r
    """
    if seed is not None:
        np.random.seed(seed)

    t_grid = np.arange(0, T, dt)
    N = len(t_grid)

    # Pre-compute Delta(t), gamma(t) (shared across all trajectories)
    t_tensor = torch.tensor(t_grid, dtype=torch.float64)
    Delta_full, Gamma_full = compute_delta_gamma(t_tensor, alpha, r, omega0, kBT)
    Delta_np = Delta_full.numpy()
    Gamma_np = Gamma_full.numpy()

    bloch_traj = np.zeros((num_traj, N, 3), dtype=np.float32)
    meas_traj = np.zeros((num_traj, N), dtype=np.float32)

    for i in range(num_traj):
# ========== 初始态生成 ==========
        x0 = random.uniform(-0.8, 0.8)
        y0 = random.uniform(-0.8, 0.8)
        z0 = random.uniform(-1.0, 1.0)

        xyz = torch.tensor([x0, y0, z0])
        norm = torch.norm(xyz)
        if norm > 1.0:
            xyz = xyz / norm  # 投影到布洛赫球面
        x0, y0, z0 = xyz.tolist()
        y0 = [x0, y0, z0]
    # ======================================

        sol = solve_ivp(
            bloch_rhs, [0, T], y0,
            t_eval=t_grid,
            args=(alpha, r, omega0, M, kBT),
            method='RK45',
            rtol=1e-8, atol=1e-10,
    )



        bloch_traj[i] = sol.y.T.astype(np.float32)
        meas_traj[i] = np.sqrt(M * zeta) * bloch_traj[i, :, 2]

    return {
        'alpha': alpha,
        'r': r,
        'bloch': torch.from_numpy(bloch_traj),
        'delta': torch.from_numpy(Delta_np[None, :].repeat(num_traj, axis=0).astype(np.float32)),
        'gamma': torch.from_numpy(Gamma_np[None, :].repeat(num_traj, axis=0).astype(np.float32)),
        'dY': torch.from_numpy(meas_traj),
        't': torch.from_numpy(t_grid.astype(np.float32)),
    }


def build_task_dataset(task_params_list, verbose=True, **kwargs):
    """Build multi-task dataset.

    Args:
        task_params_list: [(alpha1, r1), (alpha2, r2), ...]

    Returns:
        dict: {task_id: task_data_dict}
    """
    dataset = {}
    for idx, (alpha, r) in enumerate(task_params_list):
        if verbose:
            print(f"[{idx+1}/{len(task_params_list)}] alpha={alpha:.2f}, r={r:.2f} ...", end=' ')
        dataset[idx] = generate_task_data(alpha, r, **kwargs)
        if verbose:
            print(f"done, bloch shape {tuple(dataset[idx]['bloch'].shape)}")
    return dataset


if __name__ == '__main__':
    # Quick test: generate 5 trajectories for one (alpha, r)
    test_data = generate_task_data(alpha=0.5, r=0.3, num_traj=5, T=2.0, seed=42)
    print("Test data shapes:")
    print(f"  bloch: {test_data['bloch'].shape}")
    print(f"  delta: {test_data['delta'].shape}")
    print(f"  gamma: {test_data['gamma'].shape}")
    print(f"  dY:    {test_data['dY'].shape}")
    print(f"  t:     {test_data['t'].shape}")

    print(f"  Delta(t=0)  = {test_data['delta'][0, 0].item():.4f}")
    print(f"  Delta(t~T)  = {test_data['delta'][0, -1].item():.4f}")
    print(f"  gamma(t=0)  = {test_data['gamma'][0, 0].item():.4f}")
    print(f"  gamma(t~T)  = {test_data['gamma'][0, -1].item():.4f}")

    # Generate plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    t = test_data['t'].numpy()
    n_show = min(3, test_data['bloch'].shape[0])
    for i in range(n_show):
        axes[0, 0].plot(t, test_data['bloch'][i, :, 0].numpy(), label=f'traj {i}')
    axes[0, 0].set_title('x(t)')
    axes[0, 0].legend()

    axes[0, 1].plot(t, test_data['delta'][0].numpy())
    axes[0, 1].set_title('Delta(t)')

    axes[1, 0].plot(t, test_data['gamma'][0].numpy())
    axes[1, 0].set_title('gamma(t)')

    axes[1, 1].plot(t, test_data['dY'][0].numpy())
    axes[1, 1].set_title('Weak measurement dY(t)')

    save_path = os.path.join(tempfile.gettempdir(), 'aqnode_data_test.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved plot to {save_path}")
    print("All checks passed!")


# models.py — Meta-AQNODE: MAML (model-agnostic meta-learning)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from bisect import bisect_left
from torchdiffeq import odeint as odeint_orig
