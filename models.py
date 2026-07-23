# models.py - Meta-AQNODE: MAML architecture
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
<<<<<<< HEAD
from scipy.integrate import solve_ivp

from data_gen import bloch_rhs, compute_delta_gamma
=======


def _tensor_to_numpy_safe(tensor, dtype=None):
    array = np.asarray(tensor.detach().cpu().tolist())
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


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
>>>>>>> 2.0


class AQNodeBase(nn.Module):
    """Base AQNode with direct eta -> (alpha, r) decoding and physics-only rollout."""

    def __init__(
        self,
        latent_dim=64,
        context_dim=8,
        measurement_dim=50,
        omega0=1.0,
        measurement_strength=0.4,
    ):
        super().__init__()
<<<<<<< HEAD
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.measurement_dim = measurement_dim
        self.output_dim = 3
        self.omega0 = omega0
        self.measurement_strength = measurement_strength
        self.alpha_min = 0.05
        self.alpha_max = 1.0
        self.r_min = 0.05
        self.r_max = 1.0

    def encode_initial(self, init_state, dY_seq):
        """The clean physics-only model needs only the initial Bloch state."""
        return init_state

    def _compute_amplitude(self, alpha, r):
        r_safe = torch.clamp(r, min=1e-6)
        r2 = r_safe ** 2
        return (alpha ** 2) * r2 / (1.0 + r2)

    def _compute_tcl_generator_from_params(self, t_grid, alpha, r):
        """Generate physically constrained Delta/gamma from task-level alpha and cutoff ratio r."""
        t = t_grid.view(-1, 1)
        alpha = alpha.view(1, -1)
        r = r.view(1, -1)
        r_safe = torch.clamp(r, min=1e-6)
        amplitude = self._compute_amplitude(alpha, r_safe)

        exp_term = torch.exp(-r_safe * self.omega0 * t)
        cos_term = torch.cos(self.omega0 * t)
        sin_term = torch.sin(self.omega0 * t)

        gamma = amplitude * self.omega0 * (
            1.0 - exp_term * cos_term - r_safe * sin_term
        )
        delta = 2.0 * amplitude * (
            1.0 - exp_term * (cos_term - sin_term / r_safe)
        )
        return delta, gamma

    def _decode_eta_to_task_params(self, eta):
        """Treat eta itself as the task parameter body: eta = [raw_alpha, raw_r]."""
        if eta.numel() != 2:
            raise ValueError(f"eta must be 2D [raw_alpha, raw_r], got shape {tuple(eta.shape)}")
        eta = eta.view(-1)
        raw_alpha = eta[0]
        raw_r = eta[1]
        pred_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_alpha)
        pred_r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw_r)
        pred_A = self._compute_amplitude(pred_alpha, pred_r)
        return {
            "pred_A": pred_A,
            "pred_alpha": pred_alpha,
            "pred_r": pred_r,
        }

    def _project_to_bloch_ball(self, state):
        """Project predicted Bloch vectors back to the physical unit ball."""
        radius = torch.linalg.norm(state, dim=-1, keepdim=True)
        return state / torch.clamp(radius, min=1.0)

    def _integrate_tcl_bloch(self, init_state, delta_traj, gamma_traj, t_grid):
        """Integrate the TCL Bloch equations using formula-derived Delta/gamma."""
        bloch_states = [self._project_to_bloch_ball(init_state)]
        half_measurement = self.measurement_strength / 2.0

        for idx in range(1, t_grid.shape[0]):
            dt = t_grid[idx] - t_grid[idx - 1]
            prev = bloch_states[-1]
            delta_prev = delta_traj[idx - 1]
            gamma_prev = gamma_traj[idx - 1]

            rhs_prev = torch.stack(
                [
                    -(delta_prev + half_measurement) * prev[:, 0] - self.omega0 * prev[:, 1],
                    self.omega0 * prev[:, 0] - (delta_prev + half_measurement) * prev[:, 1],
                    -2.0 * gamma_prev - 2.0 * delta_prev * prev[:, 2],
                ],
                dim=-1,
            )
            proposal = prev + dt * rhs_prev

            delta_next = delta_traj[idx]
            gamma_next = gamma_traj[idx]
            rhs_next = torch.stack(
                [
                    -(delta_next + half_measurement) * proposal[:, 0] - self.omega0 * proposal[:, 1],
                    self.omega0 * proposal[:, 0] - (delta_next + half_measurement) * proposal[:, 1],
                    -2.0 * gamma_next - 2.0 * delta_next * proposal[:, 2],
                ],
                dim=-1,
            )
            next_state = prev + 0.5 * dt * (rhs_prev + rhs_next)
            bloch_states.append(self._project_to_bloch_ball(next_state))

        return torch.stack(bloch_states, dim=0)

    def forward(
        self,
        init_state,
        t_grid,
        eta,
        return_aux=False,
    ):
        t_grid = t_grid.to(init_state.device)
        aux = self._decode_eta_to_task_params(eta)
        pred_alpha = aux["pred_alpha"]
        pred_r = aux["pred_r"]
        base_delta, base_gamma = self._compute_tcl_generator_from_params(
            t_grid,
            pred_alpha.unsqueeze(0),
            pred_r.unsqueeze(0),
        )
        batch_size = init_state.shape[0]
        base_delta = base_delta.repeat(1, batch_size)
        base_gamma = base_gamma.repeat(1, batch_size)
        pred_delta = base_delta
        pred_gamma = base_gamma
        pred_bloch = self._integrate_tcl_bloch(init_state, pred_delta, pred_gamma, t_grid)
        aux["pred_delta_traj"] = pred_delta
        aux["pred_gamma_traj"] = pred_gamma
        if return_aux:
            return pred_bloch, aux
        return pred_bloch
=======
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.measurement_dim = measurement_dim
        self.output_dim = 3
        self.omega0 = omega0
        self.measurement_strength = measurement_strength
        self.alpha_min = 0.05
        self.alpha_max = 1.0
        self.r_min = 0.05
        self.r_max = 1.0

    def encode_initial(self, init_state, _dY_seq):
        """The clean physics-only model needs only the initial Bloch state."""
        return init_state

    def _compute_amplitude(self, alpha, r):
        r_safe = torch.clamp(r, min=1e-6)
        r2 = r_safe ** 2
        return (alpha ** 2) * r2 / (1.0 + r2)

    def _compute_tcl_generator_from_params(self, t_grid, alpha, r):
        """Generate physically constrained Delta/gamma from task-level alpha and cutoff ratio r."""
        t = t_grid.view(-1, 1)
        alpha = alpha.view(1, -1)
        r = r.view(1, -1)
        r_safe = torch.clamp(r, min=1e-6)
        amplitude = self._compute_amplitude(alpha, r_safe)

        exp_term = torch.exp(-r_safe * self.omega0 * t)
        cos_term = torch.cos(self.omega0 * t)
        sin_term = torch.sin(self.omega0 * t)

        gamma = amplitude * self.omega0 * (
            1.0 - exp_term * cos_term - r_safe * sin_term
        )
        delta = 2.0 * amplitude * (
            1.0 - exp_term * (cos_term - sin_term / r_safe)
        )
        return delta, gamma

    def _decode_eta_to_task_params(self, eta):
        """Treat eta itself as the task parameter body: eta = [raw_alpha, raw_r]."""
        if eta.numel() != 2:
            raise ValueError(f"eta must be 2D [raw_alpha, raw_r], got shape {tuple(eta.shape)}")
        eta = eta.view(-1)
        raw_alpha = eta[0]
        raw_r = eta[1]
        pred_alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(raw_alpha)
        pred_r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw_r)
        pred_A = self._compute_amplitude(pred_alpha, pred_r)
        return {
            "pred_A": pred_A,
            "pred_alpha": pred_alpha,
            "pred_r": pred_r,
        }

    def _project_to_bloch_ball(self, state):
        """Project predicted Bloch vectors back to the physical unit ball."""
        return _project_to_bloch_ball_tensor(state)

    def _integrate_tcl_bloch(self, init_state, delta_traj, gamma_traj, t_grid):
        """Integrate the TCL Bloch equations using formula-derived Delta/gamma."""
        return _integrate_tcl_bloch_scripted(
            init_state,
            delta_traj,
            gamma_traj,
            t_grid,
            float(self.omega0),
            float(self.measurement_strength),
        )

    def forward(
        self,
        init_state,
        t_grid,
        eta,
        return_aux=False,
    ):
        t_grid = t_grid.to(init_state.device)
        aux = self._decode_eta_to_task_params(eta)
        pred_alpha = aux["pred_alpha"]
        pred_r = aux["pred_r"]
        base_delta, base_gamma = self._compute_tcl_generator_from_params(
            t_grid,
            pred_alpha.unsqueeze(0),
            pred_r.unsqueeze(0),
        )
        batch_size = init_state.shape[0]
        base_delta = base_delta.repeat(1, batch_size)
        base_gamma = base_gamma.repeat(1, batch_size)
        pred_delta = base_delta
        pred_gamma = base_gamma
        pred_bloch = self._integrate_tcl_bloch(init_state, pred_delta, pred_gamma, t_grid)
        aux["pred_delta_traj"] = pred_delta
        aux["pred_gamma_traj"] = pred_gamma
        if return_aux:
            return pred_bloch, aux
        return pred_bloch


class InnerAQNode(nn.Module):
    """AQNode whose inner-loop variable eta is the task parameter body [raw_alpha, raw_r]."""

    def __init__(
        self,
        latent_dim=64,
        context_dim=8,
        measurement_dim=50,
        omega0=1.0,
        measurement_strength=0.4,
    ):
        super().__init__()
        if context_dim != 2:
            raise ValueError(f"context_dim must be 2 for eta=[raw_alpha, raw_r], got {context_dim}")
        self.base = AQNodeBase(
            latent_dim,
            context_dim,
            measurement_dim,
            omega0=omega0,
            measurement_strength=measurement_strength,
        )
        self.eta_init = nn.Parameter(torch.zeros(2))
        self.support_initial_window = 8
        self.support_raw_input_dim = 5
        self.support_dynamics_input_dim = 4
        self.support_initial_feat_dim = 16
        self.support_local_channels = 48
        self.support_temporal_hidden = 24
        self.support_temporal_feat_dim = 2 * self.support_temporal_hidden
        self.support_dynamics_feat_dim = (
            self.support_local_channels + self.support_temporal_feat_dim
        )
        self.support_traj_embed_dim = (
            self.support_initial_feat_dim
            + self.support_dynamics_feat_dim
        )
        self.support_initial_encoder = nn.Sequential(
            nn.Linear(self.support_initial_window + 3, 128),
            nn.ReLU(),
            nn.Linear(128, self.support_initial_feat_dim),
            nn.ReLU(),
        )
        self.support_local_encoder_branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.support_dynamics_input_dim, 16, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                for kernel_size in (3, 5, 9)
            ]
        )
        self.support_temporal_encoder = nn.GRU(
            input_size=self.support_dynamics_input_dim,
            hidden_size=self.support_temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.support_temporal_attention = nn.Sequential(
            nn.Linear(self.support_temporal_feat_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.support_traj_encoder = nn.Sequential(
            nn.Linear(self.support_traj_embed_dim, self.support_traj_embed_dim),
            nn.ReLU(),
        )
        self.support_attention = nn.Sequential(
            nn.Linear(self.support_traj_embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.support_initializer = nn.Sequential(
            nn.Linear(self.support_traj_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        self.support_shape_summary_dim = 10
        self.support_amp_aux_head = nn.Sequential(
            nn.Linear(self.support_traj_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.support_shape_aux_head = nn.Sequential(
            nn.Linear(self.support_traj_embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.support_shape_summary_dim),
        )
        self.latent_dim = latent_dim
        self.context_dim = 2

    def clone(self):
        clone = InnerAQNode(
            self.latent_dim,
            self.context_dim,
            self.base.measurement_dim,
            omega0=self.base.omega0,
            measurement_strength=self.base.measurement_strength,
        )
        clone.load_state_dict(self.state_dict())
        return clone
>>>>>>> 2.0

    def encode_initial(self, init_state, _dY_seq):
        return self.base.encode_initial(init_state, _dY_seq)

    def _build_support_trajectory_inputs(self, support_batch):
        dY = support_batch["dY"].squeeze(-1).transpose(0, 1)
        centered_dY = dY - dY[:, :1]
        if dY.shape[1] > 1:
            dy_diff = dY[:, 1:] - dY[:, :-1]
            dy_diff = torch.cat([torch.zeros_like(dy_diff[:, :1]), dy_diff], dim=1)
            dy_diff2 = dy_diff[:, 1:] - dy_diff[:, :-1]
            dy_diff2 = torch.cat([torch.zeros_like(dy_diff2[:, :1]), dy_diff2], dim=1)
        else:
            dy_diff = torch.zeros_like(dY)
            dy_diff2 = torch.zeros_like(dY)
        dy_energy = dY ** 2
        raw_inputs = torch.stack([dY, dy_diff, dy_diff2, dy_energy, centered_dY], dim=-1)
        dynamics_inputs = torch.stack([centered_dY, dy_diff, dy_diff2, dy_energy], dim=-1)
        return {
            "raw_inputs": raw_inputs,
            "dynamics_inputs": dynamics_inputs,
            "raw_dY": dY,
        }

<<<<<<< HEAD
class InnerAQNode(nn.Module):
    """AQNode whose inner-loop variable eta is the task parameter body [raw_alpha, raw_r]."""

    def __init__(
        self,
        latent_dim=64,
        context_dim=8,
        measurement_dim=50,
        omega0=1.0,
        measurement_strength=0.4,
    ):
        super().__init__()
        if context_dim != 2:
            raise ValueError(f"context_dim must be 2 for eta=[raw_alpha, raw_r], got {context_dim}")
        self.base = AQNodeBase(
            latent_dim,
            context_dim,
            measurement_dim,
            omega0=omega0,
            measurement_strength=measurement_strength,
        )
        self.eta_init = nn.Parameter(torch.zeros(2))
        self.support_initializer = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )
        self.latent_dim = latent_dim
        self.context_dim = 2

    def clone(self):
        clone = InnerAQNode(
            self.latent_dim,
            self.context_dim,
            self.base.measurement_dim,
            omega0=self.base.omega0,
            measurement_strength=self.base.measurement_strength,
        )
        clone.load_state_dict(self.state_dict())
        return clone

    def encode_initial(self, init_state, dY_seq):
        return self.base.encode_initial(init_state, dY_seq)

    def _summarize_support_batch(self, support_batch):
        init_state = support_batch["init_state"]
        dY = support_batch["dY"].squeeze(-1)
        init_mean = init_state.mean(dim=0)
        init_std = init_state.std(dim=0, unbiased=False)
        dy_mean = dY.mean().unsqueeze(0)
        dy_std = dY.std(unbiased=False).unsqueeze(0)
        return torch.cat([init_mean, init_std, dy_mean, dy_std], dim=0)

    def infer_eta_init_from_support(self, support_batch):
        support_feat = self._summarize_support_batch(support_batch)
        eta_bias = 0.1 * torch.tanh(self.support_initializer(support_feat))
        return self.eta_init + eta_bias

=======
    def _encode_initial_branch(self, raw_dY):
        prefix = raw_dY[:, :self.support_initial_window]
        if prefix.shape[1] < self.support_initial_window:
            pad = self.support_initial_window - prefix.shape[1]
            prefix = F.pad(prefix, (0, pad))
        prefix_mean = prefix.mean(dim=1, keepdim=True)
        prefix_std = prefix.std(dim=1, unbiased=False, keepdim=True)
        first_value = raw_dY[:, :1]
        initial_inputs = torch.cat([prefix, first_value, prefix_mean, prefix_std], dim=-1)
        return self.support_initial_encoder(initial_inputs)

    def _encode_dynamics_branch(self, dynamics_inputs):
        local_feat = torch.cat(
            [branch(dynamics_inputs.transpose(1, 2)).squeeze(-1) for branch in self.support_local_encoder_branches],
            dim=-1,
        )
        temporal_out, _ = self.support_temporal_encoder(dynamics_inputs)
        temporal_attn_logits = self.support_temporal_attention(temporal_out).squeeze(-1)
        temporal_attn_weights = torch.softmax(temporal_attn_logits, dim=1)
        temporal_feat = torch.sum(temporal_attn_weights.unsqueeze(-1) * temporal_out, dim=1)
        dynamics_feat = torch.cat([local_feat, temporal_feat], dim=-1)
        return dynamics_feat, temporal_attn_weights, temporal_attn_logits

    def _encode_support_set(self, support_batch, return_details=False):
        traj_inputs = self._build_support_trajectory_inputs(support_batch)
        initial_feat = self._encode_initial_branch(traj_inputs["raw_dY"])
        dynamics_feat, temporal_attn_weights, temporal_attn_logits = self._encode_dynamics_branch(
            traj_inputs["dynamics_inputs"]
        )
        traj_embed = self.support_traj_encoder(
            torch.cat([initial_feat, dynamics_feat], dim=-1)
        )
        attn_logits = self.support_attention(traj_embed).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=0)
        task_ctx = torch.sum(attn_weights.unsqueeze(-1) * traj_embed, dim=0)
        if return_details:
            branch_details = {
                "support_temporal_attn_weights": temporal_attn_weights,
                "support_temporal_attn_logits": temporal_attn_logits,
            }
            return task_ctx, attn_weights, traj_embed, attn_logits, branch_details
        return task_ctx, attn_weights

    def _summarize_traj_embeddings(self, traj_embed):
        if traj_embed.shape[0] < 2:
            return {
                "traj_embed_pairwise_cos_mean": None,
                "traj_embed_norm_mean": float(torch.norm(traj_embed, dim=-1).mean().item()),
                "traj_embed_norm_std": 0.0,
            }

        traj_embed_norm = F.normalize(traj_embed, p=2, dim=-1)
        sim_matrix = traj_embed_norm @ traj_embed_norm.transpose(0, 1)
        mask = torch.triu(torch.ones_like(sim_matrix, dtype=torch.bool), diagonal=1)
        pairwise_vals = sim_matrix[mask]
        traj_norms = torch.norm(traj_embed, dim=-1)
        return {
            "traj_embed_pairwise_cos_mean": float(pairwise_vals.mean().item()) if pairwise_vals.numel() else None,
            "traj_embed_norm_mean": float(traj_norms.mean().item()),
            "traj_embed_norm_std": float(traj_norms.std(unbiased=False).item()),
        }

    def _predict_support_physics(self, support_ctx):
        pred_support_A = F.softplus(self.support_amp_aux_head(support_ctx).squeeze(-1))
        pred_support_shape = self.support_shape_aux_head(support_ctx)
        return pred_support_A, pred_support_shape

    def infer_eta_init_from_support(self, support_batch, return_ctx=False):
        support_ctx, _ = self._encode_support_set(support_batch)
        eta_bias = 0.1 * self.support_initializer(support_ctx)
        eta0_task = self.eta_init + eta_bias
        if return_ctx:
            return eta0_task, support_ctx
        return eta0_task

>>>>>>> 2.0
    def forward(
        self,
        init_state,
        t_grid,
        eta=None,
        return_aux=False,
    ):
        if eta is None:
            eta = self.eta_init
        return self.base.forward(
            init_state,
            t_grid,
            eta,
            return_aux,
        )

    def compute_loss(
        self,
        pred,
        bloch_target,
        w_x=1.0,
        w_y=1.0,
        w_z=1.5,
    ):
        """Weighted reconstruction loss on the Bloch vector [x, y, z]."""
        loss_x = torch.mean((pred[:, :, 0] - bloch_target[:, :, 0]) ** 2)
        loss_y = torch.mean((pred[:, :, 1] - bloch_target[:, :, 1]) ** 2)
        loss_z = torch.mean((pred[:, :, 2] - bloch_target[:, :, 2]) ** 2)

        return (
            w_x * loss_x
            + w_y * loss_y
            + w_z * loss_z
        )


class MetaTrainer:
    """MAML trainer for AQNode with explicit support/query batches."""

    def __init__(
        self,
        task_dataset,
        latent_dim=64,
        outer_lr=1e-3,
        weight_decay=1e-5,
        seq_len=100,
        batch_size=20,
        w_x=1.0,
        w_y=1.0,
        w_z=1.5,
<<<<<<< HEAD
        w_alpha=1.0,
        w_r=1.0,
=======
        w_delta=0.7,
        w_gamma=1.2,
        w_alpha=1.0,
        w_r=1.0,
        w_alpha_frac=1.0,
        w_r_frac=1.0,
>>>>>>> 2.0
        w_dy=0.0,
        inner_steps=5,
        inner_lr=0.1,
        context_dim=8,
        eta_reg_weight=1e-3,
        device="cuda:0",
        measurement_dim=None,
<<<<<<< HEAD
        resimulate_query=True,
        omega0=1.0,
        measurement_strength=0.4,
=======
        omega0=1.0,
        measurement_strength=0.4,
        param_mae_weight=0.5,
        amplitude_loss_weight=0.25,
        support_amp_aux_weight=0.1,
        support_shape_aux_weight=0.1,
        support_contrastive_weight=0.1,
        support_contrastive_amp_scale=0.003,
        support_contrastive_param_threshold=0.12,
        support_contrastive_target_max_sim=0.9,
        support_contrastive_topk=8,
        lambda_smooth=0.1,
        lambda_lat=0.0,
        lambda_dist=0.0,
        val_param_weight=1.0,
        val_ood_mse_weight=1.0,
        val_bloch_weight=0.25,
>>>>>>> 2.0
        w_a=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_dataset = task_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.inner_steps = inner_steps
<<<<<<< HEAD
        self.inner_lr = inner_lr
=======
        init_inner_lr = torch.full((2,), float(inner_lr), dtype=torch.float32, device=self.device)
        self.raw_inner_lr = nn.Parameter(torch.log(torch.expm1(init_inner_lr)))
>>>>>>> 2.0
        self.context_dim = context_dim
        self.eta_reg_weight = eta_reg_weight
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z
<<<<<<< HEAD
        self.w_alpha = w_alpha if w_a is None else w_a
        self.w_r = w_r
        self.w_dy = w_dy
        self.measurement_dim = measurement_dim if measurement_dim is not None else seq_len
        self.resimulate_query = resimulate_query
        self.omega0 = omega0
        self.measurement_strength = measurement_strength
=======
        self.w_delta = w_delta
        self.w_gamma = w_gamma
        self.w_alpha = w_alpha if w_a is None else w_a
        self.w_r = w_r
        self.w_alpha_frac = w_alpha_frac
        self.w_r_frac = w_r_frac
        self.w_dy = w_dy
        self.measurement_dim = measurement_dim if measurement_dim is not None else seq_len
        self.omega0 = omega0
        self.measurement_strength = measurement_strength
        self.param_mae_weight = param_mae_weight
        self.amplitude_loss_weight = amplitude_loss_weight
        self.support_amp_aux_weight = support_amp_aux_weight
        self.support_shape_aux_weight = support_shape_aux_weight
        self.support_contrastive_weight = support_contrastive_weight
        self.support_contrastive_amp_scale = support_contrastive_amp_scale
        self.support_contrastive_param_threshold = support_contrastive_param_threshold
        self.support_contrastive_target_max_sim = support_contrastive_target_max_sim
        self.support_contrastive_topk = support_contrastive_topk
        self.lambda_smooth = lambda_smooth
        self.lambda_lat = lambda_lat
        self.lambda_dist = lambda_dist
        self.val_param_weight = val_param_weight
        self.val_ood_mse_weight = val_ood_mse_weight
        self.val_bloch_weight = val_bloch_weight
>>>>>>> 2.0

        self.meta_net = InnerAQNode(
            latent_dim,
            context_dim=context_dim,
            measurement_dim=self.measurement_dim,
            omega0=omega0,
            measurement_strength=measurement_strength,
        ).to(self.device)
        self.best_net = copy.deepcopy(self.meta_net)
        self.best_raw_inner_lr = self.raw_inner_lr.detach().clone()
        self.lowest_loss = float("inf")

        self.optimizer = optim.Adam(
            list(self.meta_net.parameters()) + [self.raw_inner_lr],
            lr=outer_lr,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=500, gamma=0.5)
        self.metrics_his = []
        self.val_metrics_his = []
        self.selection_score_his = []
        self.train_recon_his = []
        self.train_param_his = []
        self.train_eta_shift_his = []
        self.train_eta_norm_his = []
<<<<<<< HEAD
        self.val_recon_his = []
        self.val_param_his = []
=======
        self.inner_lr_alpha_his = []
        self.inner_lr_r_his = []
        self.val_recon_his = []
        self.val_param_his = []
        self.val_param_mae_his = []
        self.val_worst_axis_mse_his = []
        self.train_history = []

    def _current_inner_lr(self):
        return torch.nn.functional.softplus(self.raw_inner_lr) + 1e-6
>>>>>>> 2.0

    def _compute_recon_loss(self, model, pred, batch):
        return model.compute_loss(
            pred,
            batch["bloch"],
            w_x=self.w_x,
            w_y=self.w_y,
            w_z=self.w_z,
        )

    def _compute_eta_reg(self, eta, eta_ref=None):
        if eta_ref is None:
            eta_ref = self.meta_net.eta_init
        return torch.mean((eta - eta_ref) ** 2)

<<<<<<< HEAD
    def _compute_total_loss(
        self,
        model,
        pred,
        batch,
        eta=None,
        eta_ref=None,
        include_eta_reg=False,
        aux=None,
        include_param_loss=True,
    ):
        recon_loss = self._compute_recon_loss(model, pred, batch)
        dy_loss = torch.tensor(0.0, device=self.device)
        eta_reg = torch.tensor(0.0, device=self.device)
        param_loss = torch.tensor(0.0, device=self.device)
=======
    def _build_normalized_generator_shape_summary(self, batch, amplitude):
        eps = 1e-6
        delta = batch["delta"].mean(dim=1)
        gamma = batch["gamma"].mean(dim=1)
        amp = torch.clamp(amplitude, min=eps)
        delta_norm = delta / torch.clamp(2.0 * amp, min=eps)
        gamma_norm = gamma / torch.clamp(self.omega0 * amp, min=eps)
        num_time = delta_norm.shape[0]
        sample_idx = torch.linspace(
            0,
            max(num_time - 1, 0),
            steps=self.meta_net.support_shape_summary_dim // 2,
            device=delta_norm.device,
        ).round().long()
        return torch.cat([delta_norm[sample_idx], gamma_norm[sample_idx]], dim=0)

    def _compute_support_physics_aux_terms(self, support_batch):
        support_ctx, _ = self.meta_net._encode_support_set(support_batch)
        pred_support_A, pred_support_shape = self.meta_net._predict_support_physics(support_ctx)
        target_A = self.meta_net.base._compute_amplitude(
            support_batch["task_alpha"],
            support_batch["task_r"],
        )
        target_shape = self._build_normalized_generator_shape_summary(support_batch, target_A)
        support_amp_aux = (pred_support_A - target_A) ** 2
        support_shape_aux = torch.mean((pred_support_shape - target_shape) ** 2)
        return {
            "support_amp_aux": support_amp_aux,
            "support_shape_aux": support_shape_aux,
            "support_pred_A": pred_support_A,
            "support_target_A": target_A,
        }

    def _compute_support_contrastive_loss(self, support_task_embeddings, task_alphas, task_rs):
        if len(support_task_embeddings) < 2 or self.support_contrastive_weight <= 0:
            device = self.device
            zero = torch.tensor(0.0, device=device)
            return zero, {
                "hard_pair_count": 0,
                "mean_hard_pair_amp_diff": 0.0,
                "mean_hard_pair_param_dist": 0.0,
                "mean_hard_pair_cos": 0.0,
            }

        embeddings = torch.stack(support_task_embeddings, dim=0)
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        cosine_matrix = emb_norm @ emb_norm.transpose(0, 1)
        task_alphas = torch.stack(task_alphas).to(embeddings.device)
        task_rs = torch.stack(task_rs).to(embeddings.device)
        task_amp = self.meta_net.base._compute_amplitude(task_alphas, task_rs)

        alpha_scale = max(float(self.meta_net.base.alpha_max - self.meta_net.base.alpha_min), 1e-6)
        r_scale = max(float(self.meta_net.base.r_max - self.meta_net.base.r_min), 1e-6)

        pair_entries = []
        for i in range(len(support_task_embeddings)):
            for j in range(i + 1, len(support_task_embeddings)):
                amp_diff = torch.abs(task_amp[i] - task_amp[j])
                alpha_diff = torch.abs(task_alphas[i] - task_alphas[j]) / alpha_scale
                r_diff = torch.abs(task_rs[i] - task_rs[j]) / r_scale
                param_dist = torch.sqrt(alpha_diff ** 2 + r_diff ** 2)
                if float(param_dist.detach().item()) <= self.support_contrastive_param_threshold:
                    continue
                hardness = torch.exp(-amp_diff / max(self.support_contrastive_amp_scale, 1e-6)) * (
                    param_dist - self.support_contrastive_param_threshold
                )
                pair_entries.append((hardness, i, j, amp_diff, param_dist))

        if not pair_entries:
            zero = embeddings.new_tensor(0.0)
            return zero, {
                "hard_pair_count": 0,
                "mean_hard_pair_amp_diff": 0.0,
                "mean_hard_pair_param_dist": 0.0,
                "mean_hard_pair_cos": 0.0,
            }

        pair_entries.sort(key=lambda item: float(item[0].detach().item()), reverse=True)
        selected = pair_entries[: max(1, int(self.support_contrastive_topk))]
        losses = []
        amp_diffs = []
        param_dists = []
        hard_pair_cos = []
        for hardness, i, j, amp_diff, param_dist in selected:
            cosine_sim = cosine_matrix[i, j]
            losses.append(hardness * F.relu(cosine_sim - self.support_contrastive_target_max_sim) ** 2)
            amp_diffs.append(float(amp_diff.detach().item()))
            param_dists.append(float(param_dist.detach().item()))
            hard_pair_cos.append(float(cosine_sim.detach().item()))

        contrastive_loss = torch.stack(losses).mean()
        stats = {
            "hard_pair_count": len(selected),
            "mean_hard_pair_amp_diff": float(np.mean(amp_diffs)) if amp_diffs else 0.0,
            "mean_hard_pair_param_dist": float(np.mean(param_dists)) if param_dists else 0.0,
            "mean_hard_pair_cos": float(np.mean(hard_pair_cos)) if hard_pair_cos else 0.0,
        }
        return contrastive_loss, stats

    def _compute_loss_terms(self, pred, aux, batch, eta=None, eta_ref=None, support_batch=None):
        zero = pred.new_tensor(0.0)
        terms = {}

        terms["mse_x"] = torch.mean((pred[:, :, 0] - batch["bloch"][:, :, 0]) ** 2)
        terms["mse_y"] = torch.mean((pred[:, :, 1] - batch["bloch"][:, :, 1]) ** 2)
        terms["mse_z"] = torch.mean((pred[:, :, 2] - batch["bloch"][:, :, 2]) ** 2)
        terms["recon_loss"] = (
            self.w_x * terms["mse_x"]
            + self.w_y * terms["mse_y"]
            + self.w_z * terms["mse_z"]
        )

        pred_delta = aux["pred_delta_traj"]
        pred_gamma = aux["pred_gamma_traj"]
        terms["mse_delta"] = torch.mean((pred_delta - batch["delta"]) ** 2)
        terms["mse_gamma"] = torch.mean((pred_gamma - batch["gamma"]) ** 2)
        terms["hidden_param_loss"] = (
            self.w_delta * terms["mse_delta"]
            + self.w_gamma * terms["mse_gamma"]
        )

        alpha_diff = aux["pred_alpha"] - batch["task_alpha"]
        r_diff = aux["pred_r"] - batch["task_r"]
        terms["mse_alpha"] = alpha_diff ** 2
        terms["mse_r"] = r_diff ** 2
        terms["mae_alpha"] = torch.abs(alpha_diff)
        terms["mae_r"] = torch.abs(r_diff)
        pred_alpha_frac = aux["pred_alpha"] ** 2 / (1.0 + aux["pred_r"] ** 2)
        target_alpha_frac = batch["task_alpha"] ** 2 / (1.0 + batch["task_r"] ** 2)
        pred_r_frac = aux["pred_r"] ** 2 / (1.0 + aux["pred_r"] ** 2)
        target_r_frac = batch["task_r"] ** 2 / (1.0 + batch["task_r"] ** 2)
        terms["mse_alpha_frac"] = (pred_alpha_frac - target_alpha_frac) ** 2
        terms["mse_r_frac"] = (pred_r_frac - target_r_frac) ** 2
        terms["param_loss"] = (
            self.w_alpha * terms["mse_alpha"]
            + self.w_r * terms["mse_r"]
        )
        terms["reparam_loss"] = (
            self.w_alpha_frac * terms["mse_alpha_frac"]
            + self.w_r_frac * terms["mse_r_frac"]
        )
        terms["param_mae"] = (
            self.w_alpha * terms["mae_alpha"]
            + self.w_r * terms["mae_r"]
        )

        target_A = self.meta_net.base._compute_amplitude(
            batch["task_alpha"],
            batch["task_r"],
        )
        terms["amp"] = (aux["pred_A"] - target_A) ** 2

>>>>>>> 2.0
        if self.w_dy > 0:
            scale = torch.sqrt(batch["task_M"] * batch["task_zeta"])
            dy_hat = scale * pred[:, :, 2]
            dy_true = batch["dY"].squeeze(-1)
<<<<<<< HEAD
            dy_loss = self.w_dy * torch.mean((dy_hat - dy_true) ** 2)
        if aux is not None:
            if include_param_loss:
                if self.w_alpha > 0:
                    param_loss = param_loss + self.w_alpha * (aux["pred_alpha"] - batch["task_alpha"]) ** 2
                if self.w_r > 0:
                    param_loss = param_loss + self.w_r * (aux["pred_r"] - batch["task_r"]) ** 2
        if include_eta_reg and eta is not None and self.eta_reg_weight > 0:
            eta_reg = self.eta_reg_weight * self._compute_eta_reg(eta, eta_ref=eta_ref)
        total_loss = recon_loss + dy_loss + param_loss + eta_reg
        return total_loss, recon_loss.detach(), param_loss.detach(), eta_reg.detach()

    def _build_batch(self, task_data, traj_idx, t_start=0, seq_len=None):
        traj_idx = np.asarray(traj_idx, dtype=np.int64)
=======
            terms["mse_dy"] = torch.mean((dy_hat - dy_true) ** 2)
        else:
            terms["mse_dy"] = zero

        if batch["t_grid"].numel() > 1:
            dt = torch.clamp(batch["t_grid"][1:] - batch["t_grid"][:-1], min=1e-8).unsqueeze(1)
            delta_dot = (pred_delta[1:] - pred_delta[:-1]) / dt
            gamma_dot = (pred_gamma[1:] - pred_gamma[:-1]) / dt
            terms["smooth"] = torch.mean(delta_dot ** 2) + torch.mean(gamma_dot ** 2)
        else:
            terms["smooth"] = zero

        if eta is not None:
            if eta_ref is None:
                eta_ref = self.meta_net.eta_init
            terms["eta_reg"] = self._compute_eta_reg(eta, eta_ref=eta_ref)
            terms["eta_l2"] = torch.mean(eta ** 2)
        else:
            terms["eta_reg"] = zero
            terms["eta_l2"] = zero

        if support_batch is not None and self.lambda_dist > 0:
            eta_support = self.meta_net.infer_eta_init_from_support(support_batch)
            eta_query = self.meta_net.infer_eta_init_from_support(batch)
            terms["dist"] = torch.mean((eta_support - eta_query) ** 2)
        else:
            terms["dist"] = zero

        if support_batch is not None:
            terms.update(self._compute_support_physics_aux_terms(support_batch))
        else:
            terms["support_amp_aux"] = zero
            terms["support_shape_aux"] = zero
            terms["support_pred_A"] = zero
            terms["support_target_A"] = zero

        return terms

    def compute_inner_loss(self, pred, aux, support_batch, eta, eta_ref):
        terms = self._compute_loss_terms(
            pred=pred,
            aux=aux,
            batch=support_batch,
            eta=eta,
            eta_ref=eta_ref,
        )
        inner_loss = terms["param_loss"] + terms["reparam_loss"]
        return inner_loss, terms

    def compute_outer_loss(self, pred, aux, query_batch, eta, eta_ref, support_batch=None):
        terms = self._compute_loss_terms(
            pred=pred,
            aux=aux,
            batch=query_batch,
            eta=eta,
            eta_ref=eta_ref,
            support_batch=support_batch,
        )
        outer_loss = terms["param_loss"] + terms["reparam_loss"]
        return outer_loss, terms

    def _build_batch(self, task_data, traj_idx, t_start=0, seq_len=None):
        if isinstance(traj_idx, torch.Tensor):
            traj_idx = traj_idx.detach().cpu().reshape(-1).tolist()
        elif np.isscalar(traj_idx):
            traj_idx = [int(traj_idx)]
        else:
            traj_idx = [int(idx) for idx in traj_idx]
>>>>>>> 2.0
        num_time = task_data["bloch"].shape[1]
        seq_len = num_time if seq_len is None else min(seq_len, num_time)
        t_start = max(0, min(int(t_start), max(num_time - seq_len, 0)))
        sl = slice(t_start, t_start + seq_len)

        bloch = task_data["bloch"][traj_idx, sl, :].to(self.device).permute(1, 0, 2)
        delta = task_data["delta"][traj_idx, sl].to(self.device).permute(1, 0)
        gamma = task_data["gamma"][traj_idx, sl].to(self.device).permute(1, 0)
        dY = task_data["dY"][traj_idx, sl].to(self.device).unsqueeze(-1).permute(1, 0, 2)
        t_grid = task_data["t"][sl].to(self.device)

        init_state = bloch[0].clone()

        return {
            "t_grid": t_grid,
            "dY": dY,
            "bloch": bloch,
            "delta": delta,
            "gamma": gamma,
            "init_state": init_state,
            "task_alpha": torch.tensor(float(task_data["alpha"]), device=self.device),
            "task_r": torch.tensor(float(task_data["r"]), device=self.device),
            "task_M": torch.tensor(float(task_data.get("M", 0.4)), device=self.device),
            "task_zeta": torch.tensor(float(task_data.get("zeta", 0.9)), device=self.device),
            "traj_idx": traj_idx,
            "t_start": t_start,
        }

    def sample_task_data(self, task_data, seq_len=None):
        num_traj = task_data["bloch"].shape[0]
        num_time = task_data["bloch"].shape[1]
        batch_size = min(self.batch_size, num_traj)
        seq_len = self.seq_len if seq_len is None else seq_len
        seq_len = min(seq_len, num_time)

        traj_idx = np.random.choice(num_traj, batch_size, replace=False)
        max_start = max(num_time - seq_len, 0)
        t_start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        return self._build_batch(
            task_data,
            traj_idx=traj_idx,
            t_start=t_start,
            seq_len=seq_len,
        )

    def _resimulate_batch_from_init(self, task_data, batch):
        """Regenerate batch ground truth from the current batch initial states."""
        t_grid = batch["t_grid"] - batch["t_grid"][0]
        t_np = t_grid.detach().cpu().numpy().astype(np.float64)
        num_steps = len(t_np)
        batch_size = batch["init_state"].shape[0]

        alpha = float(task_data["alpha"])
        r = float(task_data["r"])
        omega0 = float(task_data.get("omega0", 1.0))
        M = float(task_data.get("M", 0.4))
        zeta = float(task_data.get("zeta", 0.9))
        kBT = float(task_data.get("kBT", 10.0))

        bloch = np.zeros((num_steps, batch_size, 3), dtype=np.float32)
        dY = np.zeros((num_steps, batch_size, 1), dtype=np.float32)

        init_state_np = batch["init_state"].detach().cpu().numpy()
        for idx in range(batch_size):
            y0 = init_state_np[idx].astype(np.float64).tolist()
            sol = solve_ivp(
                bloch_rhs,
                [float(t_np[0]), float(t_np[-1])],
                y0,
                t_eval=t_np,
                args=(alpha, r, omega0, M, kBT),
                method="RK45",
                rtol=1e-8,
                atol=1e-10,
            )
            traj = sol.y.T.astype(np.float32)
            bloch[:, idx, :] = traj
            dY[:, idx, 0] = np.sqrt(M * zeta) * traj[:, 2]

        delta_t, gamma_t = compute_delta_gamma(
            torch.from_numpy(t_np),
            alpha,
            r,
            omega0=omega0,
            kBT=kBT,
        )
        delta = delta_t.float().unsqueeze(1).repeat(1, batch_size).to(self.device)
        gamma = gamma_t.float().unsqueeze(1).repeat(1, batch_size).to(self.device)
        dY_tensor = torch.from_numpy(dY).to(self.device)

        batch["t_grid"] = t_grid
        batch["bloch"] = torch.from_numpy(bloch).to(self.device)
        batch["delta"] = delta
        batch["gamma"] = gamma
        batch["dY"] = dY_tensor
        batch["task_M"] = torch.tensor(float(M), device=self.device)
        batch["task_zeta"] = torch.tensor(float(zeta), device=self.device)
        return batch

    def _sample_support_query(self, task_data):
        support = self.sample_task_data(task_data)
        query = self.sample_task_data(task_data)
        for _ in range(3):
            same_traj = np.array_equal(support["traj_idx"], query["traj_idx"])
            same_window = support["t_start"] == query["t_start"]
            if not (same_traj and same_window):
                break
            query = self.sample_task_data(task_data)
<<<<<<< HEAD
        if self.resimulate_query:
            support = self._resimulate_batch_from_init(task_data, support)
            query = self._resimulate_batch_from_init(task_data, query)
=======
>>>>>>> 2.0
        return support, query

    def _deterministic_support_query(self, task_data, seq_len=None):
        num_traj = task_data["bloch"].shape[0]
        support_count = max(1, min(self.batch_size, num_traj // 2))
        query_start = support_count
        query_end = min(query_start + support_count, num_traj)

        support_idx = np.arange(support_count)
        query_idx = np.arange(query_start, query_end)
        if len(query_idx) == 0:
            query_idx = support_idx

        seq_len = self.seq_len if seq_len is None else seq_len
        support = self._build_batch(
            task_data,
            traj_idx=support_idx,
            t_start=0,
            seq_len=seq_len,
        )
        query = self._build_batch(
            task_data,
            traj_idx=query_idx,
            t_start=0,
            seq_len=seq_len,
        )
        if self.resimulate_query:
            support = self._resimulate_batch_from_init(task_data, support)
            query = self._resimulate_batch_from_init(task_data, query)
        return support, query

    def _adapt_eta(self, support_batch):
        init_state_inner = self.meta_net.encode_initial(
            support_batch["init_state"],
            None,
        ).detach()

<<<<<<< HEAD
        eta0_task = self.meta_net.infer_eta_init_from_support(support_batch)
        adapted_eta = eta0_task
        _, aux_init = self.meta_net(
            init_state_inner,
            support_batch["t_grid"],
            eta=adapted_eta,
            return_aux=True,
        )

        for step_idx in range(self.inner_steps):
=======
        eta0_task, support_ctx = self.meta_net.infer_eta_init_from_support(
            support_batch,
            return_ctx=True,
        )
        adapted_eta = eta0_task

        for _ in range(self.inner_steps):
>>>>>>> 2.0
            pred, aux = self.meta_net(
                init_state_inner,
                support_batch["t_grid"],
                eta=adapted_eta,
                return_aux=True,
            )
<<<<<<< HEAD
            loss, _, _, _ = self._compute_total_loss(
                self.meta_net,
=======
            loss, _ = self.compute_inner_loss(
>>>>>>> 2.0
                pred,
                aux,
                support_batch,
                eta=adapted_eta,
                eta_ref=eta0_task,
<<<<<<< HEAD
                include_eta_reg=True,
                aux=aux,
                include_param_loss=False,
            )
            eta_grad = torch.autograd.grad(loss, adapted_eta, retain_graph=False, create_graph=False)[0]
            adapted_eta = adapted_eta - self.inner_lr * eta_grad

        return adapted_eta, eta0_task
=======
            )
            eta_grad = torch.autograd.grad(loss, adapted_eta, retain_graph=False, create_graph=False)[0]
            adapted_eta = adapted_eta - self._current_inner_lr() * eta_grad

        return adapted_eta, eta0_task, support_ctx
>>>>>>> 2.0

    def _clear_inner_loop_grads(self):
        self.meta_net.base.zero_grad()
        self.meta_net.eta_init.grad = None

    def meta_update(self, task_data):
        """Build one task-level meta objective without stepping the optimiser."""
        support_batch, query_batch = self._sample_support_query(task_data)
        self.meta_net.train()

<<<<<<< HEAD
        adapted_eta, eta0_task = self._adapt_eta(support_batch)
=======
        adapted_eta, eta0_task, support_ctx = self._adapt_eta(support_batch)
>>>>>>> 2.0

        init_state_query = self.meta_net.encode_initial(
            query_batch["init_state"],
            None,
        )
        pred_query, query_aux = self.meta_net(
            init_state_query,
            query_batch["t_grid"],
            eta=adapted_eta,
            return_aux=True,
        )
<<<<<<< HEAD
        meta_loss, recon_loss, param_loss, eta_reg = self._compute_total_loss(
            self.meta_net,
=======
        inner_loss, _ = self.compute_inner_loss(
            *self.meta_net(
                self.meta_net.encode_initial(
                    support_batch["init_state"],
                    None,
                ).detach(),
                support_batch["t_grid"],
                eta=adapted_eta,
                return_aux=True,
            ),
            support_batch=support_batch,
            eta=adapted_eta,
            eta_ref=eta0_task,
        )
        meta_loss, outer_terms = self.compute_outer_loss(
>>>>>>> 2.0
            pred_query,
            query_aux,
            query_batch,
            eta=adapted_eta,
            eta_ref=eta0_task,
<<<<<<< HEAD
            include_eta_reg=True,
            aux=query_aux,
        )

        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        eta_after = self.meta_net.eta_init.detach().clone().cpu().numpy()
        eta_shift = float(torch.norm(adapted_eta.detach().cpu() - eta0_task.detach().cpu()).item())

        return {
            "loss": meta_loss.item(),
            "recon_loss": float(recon_loss.item()),
            "param_loss": float(param_loss.item()),
            "eta_reg": float(eta_reg.item()),
            "adapted_eta": adapted_eta.detach().cpu().numpy(),
            "eta0_task": eta0_task.detach().cpu().numpy(),
            "meta_eta": eta_after,
            "eta_shift": eta_shift,
=======
            support_batch=support_batch,
        )

        current_meta_eta = _tensor_to_numpy_safe(self.meta_net.eta_init.detach().clone(), dtype=float)
        eta_shift = float(torch.norm(adapted_eta.detach().cpu() - eta0_task.detach().cpu()).item())

        return {
            "meta_loss": meta_loss,
            "loss": float(meta_loss.detach().item()),
            "inner_loss": float(inner_loss.item()),
            "recon_loss": float(outer_terms["recon_loss"].item()),
            "hidden_param_loss": float(outer_terms["hidden_param_loss"].item()),
            "param_loss": float(outer_terms["param_loss"].item()),
            "reparam_loss": float(outer_terms["reparam_loss"].item()),
            "eta_reg": float(outer_terms["eta_reg"].item()),
            "eta_l2": float(outer_terms["eta_l2"].item()),
            "param_mae": float(outer_terms["param_mae"].item()),
            "amplitude_loss": float(outer_terms["amp"].item()),
            "support_amp_aux_loss": float(outer_terms["support_amp_aux"].item()),
            "support_shape_aux_loss": float(outer_terms["support_shape_aux"].item()),
            "dy_loss": float(outer_terms["mse_dy"].item()),
            "smooth_loss": float(outer_terms["smooth"].item()),
            "dist_loss": float(outer_terms["dist"].item()),
            "mse_delta": float(outer_terms["mse_delta"].item()),
            "mse_gamma": float(outer_terms["mse_gamma"].item()),
            "mse_alpha": float(outer_terms["mse_alpha"].item()),
            "mse_r": float(outer_terms["mse_r"].item()),
            "mse_alpha_frac": float(outer_terms["mse_alpha_frac"].item()),
            "mse_r_frac": float(outer_terms["mse_r_frac"].item()),
            "adapted_eta": _tensor_to_numpy_safe(adapted_eta, dtype=float),
            "eta0_task": _tensor_to_numpy_safe(eta0_task, dtype=float),
            "meta_eta": current_meta_eta,
            "eta_shift": eta_shift,
            "support_task_embedding": support_ctx,
            "task_alpha": support_batch["task_alpha"],
            "task_r": support_batch["task_r"],
>>>>>>> 2.0
        }

    def _evaluate_batches(self, support_batch, query_batch):
        self.meta_net.eval()
<<<<<<< HEAD
        adapted_eta, eta0_task = self._adapt_eta(support_batch)

        with torch.no_grad():
=======
        adapted_eta, eta0_task, _ = self._adapt_eta(support_batch)

        with torch.no_grad():
            _, attention_weights, traj_embed, attn_logits, branch_details = self.meta_net._encode_support_set(
                support_batch,
                return_details=True,
            )
            embed_stats = self.meta_net._summarize_traj_embeddings(traj_embed)
>>>>>>> 2.0
            init_state_query = self.meta_net.encode_initial(
                query_batch["init_state"],
                None,
            )
            pred, query_aux = self.meta_net(
                init_state_query,
                query_batch["t_grid"],
                eta=adapted_eta,
                return_aux=True,
            )
<<<<<<< HEAD
            total_loss, recon_loss, param_loss, eta_reg = self._compute_total_loss(
                self.meta_net,
=======
            total_loss, outer_terms = self.compute_outer_loss(
>>>>>>> 2.0
                pred,
                query_aux,
                query_batch,
                eta=adapted_eta,
                eta_ref=eta0_task,
<<<<<<< HEAD
                include_eta_reg=True,
                aux=query_aux,
=======
                support_batch=support_batch,
>>>>>>> 2.0
            )
            bloch_mse = torch.mean((pred[:, :, :3] - query_batch["bloch"]) ** 2)
            mse_x = torch.mean((pred[:, :, 0] - query_batch["bloch"][:, :, 0]) ** 2)
            mse_y = torch.mean((pred[:, :, 1] - query_batch["bloch"][:, :, 1]) ** 2)
            mse_z = torch.mean((pred[:, :, 2] - query_batch["bloch"][:, :, 2]) ** 2)
            pred_delta = query_aux["pred_delta_traj"]
            pred_gamma = query_aux["pred_gamma_traj"]
            mse_delta = torch.mean((pred_delta - query_batch["delta"]) ** 2)
            mse_gamma = torch.mean((pred_gamma - query_batch["gamma"]) ** 2)
            err_alpha = torch.abs(query_aux["pred_alpha"] - query_batch["task_alpha"])
            err_r = torch.abs(query_aux["pred_r"] - query_batch["task_r"])
<<<<<<< HEAD
=======
            max_axis_mse = torch.max(torch.stack([mse_x, mse_y, mse_z]))
>>>>>>> 2.0

        self.meta_net.train()
        return {
            "loss": float(total_loss.item()),
<<<<<<< HEAD
            "recon_loss": float(recon_loss.item()),
            "param_loss": float(param_loss.item()),
            "eta_reg": float(eta_reg.item()),
=======
            "recon_loss": float(outer_terms["recon_loss"].item()),
            "hidden_param_loss": float(outer_terms["hidden_param_loss"].item()),
            "param_loss": float(outer_terms["param_loss"].item()),
            "param_mae": float(outer_terms["param_mae"].item()),
            "reparam_loss": float(outer_terms["reparam_loss"].item()),
            "amplitude_loss": float(outer_terms["amp"].item()),
            "support_amp_aux_loss": float(outer_terms["support_amp_aux"].item()),
            "support_shape_aux_loss": float(outer_terms["support_shape_aux"].item()),
            "eta_reg": float(outer_terms["eta_reg"].item()),
            "eta_l2": float(outer_terms["eta_l2"].item()),
            "dy_loss": float(outer_terms["mse_dy"].item()),
            "smooth_loss": float(outer_terms["smooth"].item()),
            "dist_loss": float(outer_terms["dist"].item()),
>>>>>>> 2.0
            "bloch_mse": float(bloch_mse.item()),
            "mse_x": float(mse_x.item()),
            "mse_y": float(mse_y.item()),
            "mse_z": float(mse_z.item()),
<<<<<<< HEAD
            "mse_delta": float(mse_delta.item()),
            "mse_gamma": float(mse_gamma.item()),
=======
            "max_axis_mse": float(max_axis_mse.item()),
            "mse_delta": float(mse_delta.item()),
            "mse_gamma": float(mse_gamma.item()),
            "mse_alpha": float(outer_terms["mse_alpha"].item()),
            "mse_r": float(outer_terms["mse_r"].item()),
            "mse_alpha_frac": float(outer_terms["mse_alpha_frac"].item()),
            "mse_r_frac": float(outer_terms["mse_r_frac"].item()),
>>>>>>> 2.0
            "pred_A": float(query_aux["pred_A"].item()),
            "pred_alpha": float(query_aux["pred_alpha"].item()),
            "pred_r": float(query_aux["pred_r"].item()),
            "err_alpha": float(err_alpha.item()),
            "err_r": float(err_r.item()),
            "pred": pred,
            "pred_delta": pred_delta,
            "pred_gamma": pred_gamma,
            "batch": query_batch,
<<<<<<< HEAD
            "eta": adapted_eta.detach().cpu().numpy(),
            "eta0_task": eta0_task.detach().cpu().numpy(),
=======
            "eta": _tensor_to_numpy_safe(adapted_eta, dtype=float),
            "eta0_task": _tensor_to_numpy_safe(eta0_task, dtype=float),
            "attention_weights": _tensor_to_numpy_safe(attention_weights, dtype=float),
            "attention_logit_std": float(attn_logits.std(unbiased=False).item()),
            "temporal_attention_peak_mean": float(
                branch_details["support_temporal_attn_weights"].max(dim=1).values.mean().item()
            ),
            "traj_embed_pairwise_cos_mean": embed_stats["traj_embed_pairwise_cos_mean"],
            "traj_embed_norm_mean": embed_stats["traj_embed_norm_mean"],
            "traj_embed_norm_std": embed_stats["traj_embed_norm_std"],
            "support_traj_idx": support_batch["traj_idx"].copy(),
>>>>>>> 2.0
        }

    def train(self, num_epochs=200, tasks_per_epoch=5, val_tasks=None):
        ids = sorted(self.task_dataset.keys())
<<<<<<< HEAD
        for ep in range(num_epochs):
            chosen = ids
            ep_loss = 0.0
            ep_recon = 0.0
            ep_param = 0.0
            all_eta_shift = []
            all_eta_norm = []
=======
        if not ids:
            raise ValueError("task_dataset must contain at least one training task")
        effective_tasks_per_epoch = len(ids) if tasks_per_epoch <= 0 else min(tasks_per_epoch, len(ids))
        self.epoch_time_his = []
        for ep in range(num_epochs):
            epoch_start_time = time.perf_counter()
            if effective_tasks_per_epoch >= len(ids):
                chosen = ids.copy()
                np.random.shuffle(chosen)
            else:
                chosen = np.random.choice(ids, size=effective_tasks_per_epoch, replace=False).tolist()
            ep_loss = 0.0
            ep_recon = 0.0
            ep_param = 0.0
            ep_param_mae = 0.0
            ep_eta_reg = 0.0
            all_eta_shift = []
            all_eta_norm = []
            all_eta0_norm = []
            all_eta0_shift = []
            epoch_meta_losses = []
            epoch_support_embeddings = []
            epoch_task_alphas = []
            epoch_task_rs = []
            self.optimizer.zero_grad()
>>>>>>> 2.0
            for tid in chosen:
                update_info = self.meta_update(self.task_dataset[tid])
                epoch_meta_losses.append(update_info["meta_loss"])
                epoch_support_embeddings.append(update_info["support_task_embedding"])
                epoch_task_alphas.append(update_info["task_alpha"])
                epoch_task_rs.append(update_info["task_r"])
                ep_loss += update_info["loss"]
                ep_recon += update_info["recon_loss"]
                ep_param += update_info["param_loss"]
<<<<<<< HEAD
                all_eta_shift.append(update_info["eta_shift"])
                all_eta_norm.append(float(np.linalg.norm(update_info["adapted_eta"])))
=======
                ep_param_mae += update_info["param_mae"]
                ep_eta_reg += update_info["eta_reg"]
                all_eta_shift.append(update_info["eta_shift"])
                all_eta_norm.append(float(np.linalg.norm(update_info["adapted_eta"])))
                all_eta0_norm.append(float(np.linalg.norm(update_info["eta0_task"])))
                all_eta0_shift.append(
                    float(np.linalg.norm(update_info["eta0_task"] - update_info["meta_eta"]))
                )
>>>>>>> 2.0
                print(
                    f"Ep {ep:3d} T{tid}: query_loss={update_info['loss']:.6f} "
                    f"recon={update_info['recon_loss']:.6f} "
                    f"param={update_info['param_loss']:.6f} "
<<<<<<< HEAD
=======
                    f"param_mae={update_info['param_mae']:.6f} "
                    f"amp={update_info['amplitude_loss']:.6f} "
>>>>>>> 2.0
                    f"eta_reg={update_info['eta_reg']:.6f} "
                    f"|adapted_eta|={np.linalg.norm(update_info['adapted_eta']):.4f} "
                    f"|meta_eta|={np.linalg.norm(update_info['meta_eta']):.4f} "
                    f"eta_shift={update_info['eta_shift']:.5f}"
                )

            epoch_outer_loss = torch.stack(epoch_meta_losses).mean()
            contrastive_loss, contrastive_stats = self._compute_support_contrastive_loss(
                epoch_support_embeddings,
                epoch_task_alphas,
                epoch_task_rs,
            )
            epoch_outer_loss = epoch_outer_loss + self.support_contrastive_weight * contrastive_loss
            epoch_outer_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            avg = ep_loss / len(chosen)
            avg_recon = ep_recon / len(chosen)
            avg_param = ep_param / len(chosen)
<<<<<<< HEAD
=======
            avg_param_mae = ep_param_mae / len(chosen)
            avg_eta_reg = ep_eta_reg / len(chosen)
            avg_contrastive = float(contrastive_loss.detach().item())
>>>>>>> 2.0
            self.metrics_his.append(avg)
            self.train_recon_his.append(avg_recon)
            self.train_param_his.append(avg_param)
            self.scheduler.step()

            val_avg = None
            selection_score = None
<<<<<<< HEAD
            if val_tasks is not None:
                val_loss = 0.0
                val_recon = 0.0
                val_param = 0.0
=======
            val_param_mae_avg = None
            val_worst_axis_mse = None
            if val_tasks:
                val_loss = 0.0
                val_recon = 0.0
                val_param = 0.0
                val_param_mae = 0.0
                worst_axis_values = []
>>>>>>> 2.0
                for _, task_data in val_tasks.items():
                    support_batch, query_batch = self._deterministic_support_query(task_data)
                    metrics = self._evaluate_batches(support_batch, query_batch)
                    val_loss += metrics["loss"]
                    val_recon += metrics["recon_loss"]
                    val_param += metrics["param_loss"]
<<<<<<< HEAD
=======
                    val_param_mae += 0.5 * (metrics["err_alpha"] + metrics["err_r"])
                    worst_axis_values.append(metrics["max_axis_mse"])
>>>>>>> 2.0

                val_avg = val_loss / len(val_tasks)
                val_recon_avg = val_recon / len(val_tasks)
                val_param_avg = val_param / len(val_tasks)
<<<<<<< HEAD
                # Select checkpoints by parameter identification first, with reconstruction as a tie-breaker.
                selection_score = val_param_avg + 0.1 * val_recon_avg
                self.val_metrics_his.append(val_avg)
                self.val_recon_his.append(val_recon_avg)
                self.val_param_his.append(val_param_avg)
                self.selection_score_his.append(selection_score)
                if selection_score < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
=======
                val_param_mae_avg = val_param_mae / len(val_tasks)
                val_worst_axis_mse = max(worst_axis_values) if worst_axis_values else None
                # Select checkpoints by parameter identification and OOD-like worst-axis reconstruction.
                selection_score = (
                    self.val_param_weight * val_param_mae_avg
                    + self.val_ood_mse_weight * float(np.sqrt(max(val_worst_axis_mse, 0.0)))
                    + self.val_bloch_weight * float(np.sqrt(max(val_recon_avg, 0.0)))
                )
                self.val_metrics_his.append(val_avg)
                self.val_recon_his.append(val_recon_avg)
                self.val_param_his.append(val_param_avg)
                self.val_param_mae_his.append(val_param_mae_avg)
                self.val_worst_axis_mse_his.append(val_worst_axis_mse)
                self.selection_score_his.append(selection_score)
                if selection_score < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.best_raw_inner_lr = self.raw_inner_lr.detach().clone()
>>>>>>> 2.0
                    self.lowest_loss = selection_score
            else:
                if avg < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.best_raw_inner_lr = self.raw_inner_lr.detach().clone()
                    self.lowest_loss = avg

            val_str = f"  val={val_avg:.6f}" if val_avg is not None else ""
            select_str = f"  sel={selection_score:.6f}" if selection_score is not None else ""
<<<<<<< HEAD
            eta_shift_avg = float(np.mean(all_eta_shift)) if all_eta_shift else 0.0
            eta_norm_avg = float(np.mean(all_eta_norm)) if all_eta_norm else float(torch.norm(self.meta_net.eta_init).item())
            self.train_eta_shift_his.append(eta_shift_avg)
            self.train_eta_norm_his.append(eta_norm_avg)
            print(
                f"  Avg query loss: {avg:.6f}{val_str}{select_str}  "
                f"mean_eta_norm={eta_norm_avg:.4f}  "
                f"mean_eta_shift={eta_shift_avg:.5f}"
=======
            val_param_mae_str = (
                f"  val_param_mae={val_param_mae_avg:.6f}"
                if val_param_mae_avg is not None
                else ""
            )
            val_worst_axis_str = (
                f"  val_worst_axis={val_worst_axis_mse:.6f}"
                if val_worst_axis_mse is not None
                else ""
            )
            eta_shift_avg = float(np.mean(all_eta_shift)) if all_eta_shift else 0.0
            eta_norm_avg = float(np.mean(all_eta_norm)) if all_eta_norm else float(torch.norm(self.meta_net.eta_init).item())
            eta0_norm_avg = float(np.mean(all_eta0_norm)) if all_eta0_norm else float(torch.norm(self.meta_net.eta_init).item())
            eta0_shift_avg = float(np.mean(all_eta0_shift)) if all_eta0_shift else 0.0
            current_inner_lr = _tensor_to_numpy_safe(self._current_inner_lr(), dtype=float)
            current_outer_lr = float(self.optimizer.param_groups[0]["lr"])
            self.train_eta_shift_his.append(eta_shift_avg)
            self.train_eta_norm_his.append(eta_norm_avg)
            self.inner_lr_alpha_his.append(float(current_inner_lr[0]))
            self.inner_lr_r_his.append(float(current_inner_lr[1]))
            self.train_history.append({
                "epoch": ep,
                "train_loss": avg,
                "train_recon_loss": avg_recon,
                "train_param_loss": avg_param,
                "train_param_mae": avg_param_mae,
                "train_eta_reg": avg_eta_reg,
                "train_support_contrastive_loss": avg_contrastive,
                "train_hard_pair_count": contrastive_stats["hard_pair_count"],
                "train_hard_pair_amp_diff": contrastive_stats["mean_hard_pair_amp_diff"],
                "train_hard_pair_param_dist": contrastive_stats["mean_hard_pair_param_dist"],
                "train_hard_pair_cos": contrastive_stats["mean_hard_pair_cos"],
                "val_loss": val_avg,
                "val_recon_loss": val_recon_avg if val_avg is not None else None,
                "val_param_loss": val_param_avg if val_avg is not None else None,
                "val_param_mae": val_param_mae_avg,
                "val_worst_axis_mse": val_worst_axis_mse,
                "selection_score": selection_score,
                "mean_eta_norm": eta_norm_avg,
                "mean_eta0_norm": eta0_norm_avg,
                "mean_eta_shift": eta_shift_avg,
                "mean_eta0_shift": eta0_shift_avg,
                "inner_lr_alpha": float(current_inner_lr[0]),
                "inner_lr_r": float(current_inner_lr[1]),
                "lr": current_outer_lr,
                "epoch_time_sec": float(time.perf_counter() - epoch_start_time),
            })
            epoch_time_sec = self.train_history[-1]["epoch_time_sec"]
            self.epoch_time_his.append(epoch_time_sec)
            print(
                f"  Avg query loss: {avg:.6f}  "
                f"tasks_this_epoch={len(chosen)}  "
                f"train_param_mae={avg_param_mae:.6f}  "
                f"contrastive={avg_contrastive:.6f}  "
                f"hard_pairs={contrastive_stats['hard_pair_count']}"
                f"{val_str}{val_param_mae_str}{val_worst_axis_str}{select_str}  "
                f"mean_eta_norm={eta_norm_avg:.4f}  "
                f"mean_eta0_norm={eta0_norm_avg:.4f}  "
                f"mean_eta_shift={eta_shift_avg:.5f}  "
                f"inner_lr=[{current_inner_lr[0]:.4f}, {current_inner_lr[1]:.4f}]  "
                f"epoch_time={epoch_time_sec:.2f}s"
>>>>>>> 2.0
            )

    def evaluate(self, task_data):
        self.meta_net.load_state_dict(self.best_net.state_dict())
        self.raw_inner_lr.data.copy_(self.best_raw_inner_lr)
        support_batch, query_batch = self._deterministic_support_query(task_data)
        return self._evaluate_batches(support_batch, query_batch)

    def predict_full_trajectory(self, task_data, traj_idx=0):
        self.meta_net.load_state_dict(self.best_net.state_dict())
        self.raw_inner_lr.data.copy_(self.best_raw_inner_lr)

        num_traj = task_data["bloch"].shape[0]
        support_candidates = [idx for idx in range(num_traj) if idx != traj_idx]
        if not support_candidates:
            support_candidates = [traj_idx]
        support_idx = np.asarray(support_candidates[: min(self.batch_size, len(support_candidates))])

        support_batch = self._build_batch(
            task_data,
            traj_idx=support_idx,
            t_start=0,
            seq_len=min(self.seq_len, task_data["bloch"].shape[1]),
        )
        query_batch = self._build_batch(
            task_data,
            traj_idx=np.asarray([traj_idx]),
            t_start=0,
            seq_len=task_data["bloch"].shape[1],
        )
        if self.resimulate_query:
            support_batch = self._resimulate_batch_from_init(task_data, support_batch)
            query_batch = self._resimulate_batch_from_init(task_data, query_batch)
        metrics = self._evaluate_batches(support_batch, query_batch)
<<<<<<< HEAD
        pred_full = metrics["pred"][:, 0, :].detach().cpu().numpy()
        true_traj = query_batch["bloch"][:, 0, :3].detach().cpu().numpy()
        true_delta = query_batch["delta"][:, 0].detach().cpu().numpy()
        true_gamma = query_batch["gamma"][:, 0].detach().cpu().numpy()
        pred_traj = pred_full.copy()
        pred_delta = metrics["pred_delta"][:, 0].detach().cpu().numpy()
        pred_gamma = metrics["pred_gamma"][:, 0].detach().cpu().numpy()
=======
        pred_full = _tensor_to_numpy_safe(metrics["pred"][:, 0, :], dtype=float)
        true_traj = _tensor_to_numpy_safe(query_batch["bloch"][:, 0, :3], dtype=float)
        true_delta = _tensor_to_numpy_safe(query_batch["delta"][:, 0], dtype=float)
        true_gamma = _tensor_to_numpy_safe(query_batch["gamma"][:, 0], dtype=float)
        pred_traj = pred_full.copy()
        pred_delta = _tensor_to_numpy_safe(metrics["pred_delta"][:, 0], dtype=float)
        pred_gamma = _tensor_to_numpy_safe(metrics["pred_gamma"][:, 0], dtype=float)
>>>>>>> 2.0

        return {
            "loss": metrics["loss"],
            "bloch_mse": metrics["bloch_mse"],
            "mse_x": metrics["mse_x"],
            "mse_y": metrics["mse_y"],
            "mse_z": metrics["mse_z"],
<<<<<<< HEAD
            "mse_delta": metrics["mse_delta"],
            "mse_gamma": metrics["mse_gamma"],
            "param_loss": metrics["param_loss"],
=======
            "max_axis_mse": metrics["max_axis_mse"],
            "mse_delta": metrics["mse_delta"],
            "mse_gamma": metrics["mse_gamma"],
            "param_loss": metrics["param_loss"],
            "param_mae": metrics["param_mae"],
            "amplitude_loss": metrics["amplitude_loss"],
            "support_amp_aux_loss": metrics["support_amp_aux_loss"],
            "support_shape_aux_loss": metrics["support_shape_aux_loss"],
>>>>>>> 2.0
            "pred_A": metrics["pred_A"],
            "pred_alpha": metrics["pred_alpha"],
            "pred_r": metrics["pred_r"],
            "err_alpha": metrics["err_alpha"],
            "err_r": metrics["err_r"],
            "eta": metrics["eta"],
<<<<<<< HEAD
=======
            "attention_weights": metrics["attention_weights"],
            "support_traj_idx": metrics["support_traj_idx"],
>>>>>>> 2.0
            "true_traj": true_traj,
            "pred_traj": pred_traj,
            "true_delta": true_delta,
            "pred_delta": pred_delta,
            "true_gamma": true_gamma,
            "pred_gamma": pred_gamma,
        }

    def eval_on_all_tasks(self, test_dataset):
        results = {}
        for tid, task_data in test_dataset.items():
            metrics = self.evaluate(task_data)
            results[tid] = {
                "loss": metrics["loss"],
                "bloch_mse": metrics["bloch_mse"],
                "mse_x": metrics["mse_x"],
                "mse_y": metrics["mse_y"],
                "mse_z": metrics["mse_z"],
<<<<<<< HEAD
                "mse_delta": metrics["mse_delta"],
                "mse_gamma": metrics["mse_gamma"],
                "param_loss": metrics["param_loss"],
=======
                "max_axis_mse": metrics["max_axis_mse"],
                "mse_delta": metrics["mse_delta"],
                "mse_gamma": metrics["mse_gamma"],
                "param_loss": metrics["param_loss"],
                "param_mae": metrics["param_mae"],
                "amplitude_loss": metrics["amplitude_loss"],
                "support_amp_aux_loss": metrics["support_amp_aux_loss"],
                "support_shape_aux_loss": metrics["support_shape_aux_loss"],
>>>>>>> 2.0
                "A": float(self.meta_net.base._compute_amplitude(
                    torch.tensor(float(task_data["alpha"])),
                    torch.tensor(float(task_data["r"])),
                ).item()),
                "alpha": task_data["alpha"],
                "r": task_data["r"],
                "pred_A": metrics["pred_A"],
                "pred_alpha": metrics["pred_alpha"],
                "pred_r": metrics["pred_r"],
                "err_A": abs(metrics["pred_A"] - float(self.meta_net.base._compute_amplitude(
                    torch.tensor(float(task_data["alpha"])),
                    torch.tensor(float(task_data["r"])),
                ).item())),
                "err_alpha": metrics["err_alpha"],
                "err_r": metrics["err_r"],
                "eta": metrics["eta"],
<<<<<<< HEAD
=======
                "attention_weights": metrics["attention_weights"],
                "attention_logit_std": metrics["attention_logit_std"],
                "traj_embed_pairwise_cos_mean": metrics["traj_embed_pairwise_cos_mean"],
                "traj_embed_norm_mean": metrics["traj_embed_norm_mean"],
                "traj_embed_norm_std": metrics["traj_embed_norm_std"],
                "support_traj_idx": metrics["support_traj_idx"],
>>>>>>> 2.0
            }
        return results
