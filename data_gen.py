# data_gen.py - synthetic quantum trajectory data for Meta-AQNODE
import os
import tempfile

import numpy as np
import torch


@torch.jit.script
def _project_to_bloch_ball_tensor(state: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.norm(state, dim=-1, keepdim=True)
    return state / torch.clamp_min(radius, 1.0)


@torch.jit.script
def _integrate_tcl_bloch_scripted(
    init_state: torch.Tensor,
    delta_traj: torch.Tensor,
    gamma_traj: torch.Tensor,
    t_grid: torch.Tensor,
    omega0: float,
    measurement_strength: float,
) -> torch.Tensor:
    num_steps = t_grid.size(0)
    batch_size = init_state.size(0)
    out = torch.empty((num_steps, batch_size, 3), dtype=init_state.dtype, device=init_state.device)
    prev = _project_to_bloch_ball_tensor(init_state)
    out[0] = prev
    if num_steps <= 1:
        return out

    half_measurement = measurement_strength / 2.0
    dt_all = t_grid[1:] - t_grid[:-1]

    for idx in range(num_steps - 1):
        dt = dt_all[idx]

        delta_prev = delta_traj[idx]
        gamma_prev = gamma_traj[idx]
        coeff_prev = delta_prev + half_measurement
        x_prev = prev[:, 0]
        y_prev = prev[:, 1]
        z_prev = prev[:, 2]

        rhs_prev_x = -coeff_prev * x_prev - omega0 * y_prev
        rhs_prev_y = omega0 * x_prev - coeff_prev * y_prev
        rhs_prev_z = -2.0 * gamma_prev - 2.0 * delta_prev * z_prev

        proposal_x = x_prev + dt * rhs_prev_x
        proposal_y = y_prev + dt * rhs_prev_y
        proposal_z = z_prev + dt * rhs_prev_z

        delta_next = delta_traj[idx + 1]
        gamma_next = gamma_traj[idx + 1]
        coeff_next = delta_next + half_measurement

        rhs_next_x = -coeff_next * proposal_x - omega0 * proposal_y
        rhs_next_y = omega0 * proposal_x - coeff_next * proposal_y
        rhs_next_z = -2.0 * gamma_next - 2.0 * delta_next * proposal_z

        next_x = x_prev + 0.5 * dt * (rhs_prev_x + rhs_next_x)
        next_y = y_prev + 0.5 * dt * (rhs_prev_y + rhs_next_y)
        next_z = z_prev + 0.5 * dt * (rhs_prev_z + rhs_next_z)

        prev = _project_to_bloch_ball_tensor(
            torch.stack((next_x, next_y, next_z), dim=-1)
        )
        out[idx + 1] = prev

    return out

def delta_fn(t, alpha, r, omega_0):
    """Stable scalar/numpy Delta(t) definition."""
    exp_term = np.exp(-r * omega_0 * t)
    cos_term = np.cos(omega_0 * t)
    sin_term = np.sin(omega_0 * t)

    delta = 2 * alpha ** 2 * r ** 2 / (1 + r ** 2) * (
        1 - exp_term * (cos_term - sin_term / r)
    )
    if np.isnan(delta) or np.isinf(delta):
        return 0.0
    return delta


def gamma_fn(t, alpha, r, omega_0):
    """Stable scalar/numpy gamma(t) definition."""
    exp_term = np.exp(-r * omega_0 * t)
    cos_term = np.cos(omega_0 * t)
    sin_term = np.sin(omega_0 * t)

    gamma = alpha ** 2 * omega_0 * r ** 2 / (1 + r ** 2) * (
        1 - exp_term * cos_term - r * sin_term
    )
    if np.isnan(gamma) or np.isinf(gamma):
        return 0.0
    return gamma


def delta_fn(t, alpha, r, omega_0):
    """Stable scalar/numpy Delta(t) definition."""
    exp_term = np.exp(-r * omega_0 * t)
    cos_term = np.cos(omega_0 * t)
    sin_term = np.sin(omega_0 * t)

    delta = 2 * alpha ** 2 * r ** 2 / (1 + r ** 2) * (
        1 - exp_term * (cos_term - sin_term / r)
    )
    if np.isnan(delta) or np.isinf(delta):
        return 0.0
    return delta


def gamma_fn(t, alpha, r, omega_0):
    """Stable scalar/numpy gamma(t) definition."""
    exp_term = np.exp(-r * omega_0 * t)
    cos_term = np.cos(omega_0 * t)
    sin_term = np.sin(omega_0 * t)

    gamma = alpha ** 2 * omega_0 * r ** 2 / (1 + r ** 2) * (
        1 - exp_term * cos_term - r * sin_term
    )
    if np.isnan(gamma) or np.isinf(gamma):
        return 0.0
    return gamma


def compute_delta_gamma(t, alpha, r, omega0=1.0, kBT=10.0):
    """Compute Delta(t) and gamma(t) using the stabilized TCL formulas.

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
    exp_term = torch.exp(-r * omega0 * t)
    cos_term = torch.cos(omega0 * t)
    sin_term = torch.sin(omega0 * t)

    gamma = (alpha ** 2 * omega0 * r2 / (1 + r2)) * (
        1 - exp_term * cos_term - r * sin_term
    )
    delta = (2 * alpha ** 2 * r2 / (1 + r2)) * (
        1 - exp_term * (cos_term - sin_term / r)
    )
    delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
    gamma = torch.nan_to_num(gamma, nan=0.0, posinf=0.0, neginf=0.0)

    return delta.float(), gamma.float()
<<<<<<< HEAD


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
=======
>>>>>>> 2.0


def weak_measurement(y, M=0.4, zeta=0.9):
    """Weak measurement signal (Eq.7).

    dY/dt = sqrt(M * zeta) * z

    Returns:
        dY/dt scalar
    """
    return np.sqrt(M * zeta) * y[2]


def generate_task_data(alpha, r, num_traj=200, T=10.0, dt=0.01,
                       omega0=1.0, M=0.4, zeta=0.9, kBT=10.0, seed=None, device=None):
    """Generate multiple trajectories for one (alpha, r) task.

    Each trajectory:
      - Random initial Bloch vector
      - Batch integrate Bloch Eq.5 with a scripted tensor solver
      - Compute weak measurement Eq.7
      - Pre-compute analytical Delta(t), gamma(t) (Eq.2-3)

    Returns:
        dict with keys: bloch, delta, gamma, dY, t, alpha, r
    """
    target_device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if seed is not None:
        np.random.seed(seed)

    t_grid = torch.arange(0, T, dt, dtype=torch.float32, device=target_device)
    num_steps = int(t_grid.numel())

    # Delta/gamma are shared across all trajectories of one task.
    delta_full, gamma_full = compute_delta_gamma(
        t_grid.to(dtype=torch.float64), alpha, r, omega0, kBT
    )
    delta_full = delta_full.to(device=target_device, dtype=torch.float32)
    gamma_full = gamma_full.to(device=target_device, dtype=torch.float32)

    generator = torch.Generator(device=target_device)
    if seed is not None:
        generator.manual_seed(seed)

    init_state = torch.empty((num_traj, 3), dtype=torch.float32, device=target_device)
    init_state[:, 0].uniform_(-0.8, 0.8, generator=generator)
    init_state[:, 1].uniform_(-0.8, 0.8, generator=generator)
    init_state[:, 2].uniform_(-1.0, 1.0, generator=generator)
    init_state = _project_to_bloch_ball_tensor(init_state)

    delta_traj = delta_full.unsqueeze(1).expand(num_steps, num_traj)
    gamma_traj = gamma_full.unsqueeze(1).expand(num_steps, num_traj)
    bloch_traj = _integrate_tcl_bloch_scripted(
        init_state=init_state,
        delta_traj=delta_traj,
        gamma_traj=gamma_traj,
        t_grid=t_grid,
        omega0=float(omega0),
        measurement_strength=float(M),
    )
    bloch_bt = bloch_traj.permute(1, 0, 2).contiguous()
    meas_traj = (float(np.sqrt(M * zeta)) * bloch_bt[:, :, 2]).contiguous()

    return {
        'alpha': alpha,
        'r': r,
        'omega0': omega0,
        'M': M,
        'zeta': zeta,
        'kBT': kBT,
<<<<<<< HEAD
        'bloch': torch.from_numpy(bloch_traj),
        'delta': torch.from_numpy(Delta_np[None, :].repeat(num_traj, axis=0).astype(np.float32)),
        'gamma': torch.from_numpy(Gamma_np[None, :].repeat(num_traj, axis=0).astype(np.float32)),
        'dY': torch.from_numpy(meas_traj),
        't': torch.from_numpy(t_grid.astype(np.float32)),
=======
        'bloch': bloch_bt.cpu(),
        'delta': delta_traj.transpose(0, 1).contiguous().cpu(),
        'gamma': gamma_traj.transpose(0, 1).contiguous().cpu(),
        'dY': meas_traj.cpu(),
        't': t_grid.cpu(),
>>>>>>> 2.0
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
