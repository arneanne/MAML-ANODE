# models.py - Meta-AQNODE: MAML architecture
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.integrate import solve_ivp

from data_gen import bloch_rhs, compute_delta_gamma


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
        w_alpha=1.0,
        w_r=1.0,
        w_dy=0.0,
        inner_steps=5,
        inner_lr=0.1,
        context_dim=8,
        eta_reg_weight=1e-3,
        device="cuda:0",
        measurement_dim=None,
        resimulate_query=True,
        omega0=1.0,
        measurement_strength=0.4,
        w_a=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_dataset = task_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.context_dim = context_dim
        self.eta_reg_weight = eta_reg_weight
        self.w_x = w_x
        self.w_y = w_y
        self.w_z = w_z
        self.w_alpha = w_alpha if w_a is None else w_a
        self.w_r = w_r
        self.w_dy = w_dy
        self.measurement_dim = measurement_dim if measurement_dim is not None else seq_len
        self.resimulate_query = resimulate_query
        self.omega0 = omega0
        self.measurement_strength = measurement_strength

        self.meta_net = InnerAQNode(
            latent_dim,
            context_dim=context_dim,
            measurement_dim=self.measurement_dim,
            omega0=omega0,
            measurement_strength=measurement_strength,
        ).to(self.device)
        self.best_net = copy.deepcopy(self.meta_net)
        self.lowest_loss = float("inf")

        self.optimizer = optim.Adam(
            self.meta_net.parameters(),
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
        self.val_recon_his = []
        self.val_param_his = []

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
        if self.w_dy > 0:
            scale = torch.sqrt(batch["task_M"] * batch["task_zeta"])
            dy_hat = scale * pred[:, :, 2]
            dy_true = batch["dY"].squeeze(-1)
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
        if self.resimulate_query:
            support = self._resimulate_batch_from_init(task_data, support)
            query = self._resimulate_batch_from_init(task_data, query)
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

        eta0_task = self.meta_net.infer_eta_init_from_support(support_batch)
        adapted_eta = eta0_task
        _, aux_init = self.meta_net(
            init_state_inner,
            support_batch["t_grid"],
            eta=adapted_eta,
            return_aux=True,
        )

        for step_idx in range(self.inner_steps):
            pred, aux = self.meta_net(
                init_state_inner,
                support_batch["t_grid"],
                eta=adapted_eta,
                return_aux=True,
            )
            loss, _, _, _ = self._compute_total_loss(
                self.meta_net,
                pred,
                support_batch,
                eta=adapted_eta,
                eta_ref=eta0_task,
                include_eta_reg=True,
                aux=aux,
                include_param_loss=False,
            )
            eta_grad = torch.autograd.grad(loss, adapted_eta, retain_graph=False, create_graph=False)[0]
            adapted_eta = adapted_eta - self.inner_lr * eta_grad

        return adapted_eta, eta0_task

    def _clear_inner_loop_grads(self):
        self.meta_net.base.zero_grad()
        self.meta_net.eta_init.grad = None

    def meta_update(self, task_data):
        """Single FOMAML update with separate support/query batches."""
        support_batch, query_batch = self._sample_support_query(task_data)
        self.meta_net.train()
        self.optimizer.zero_grad()

        adapted_eta, eta0_task = self._adapt_eta(support_batch)

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
        meta_loss, recon_loss, param_loss, eta_reg = self._compute_total_loss(
            self.meta_net,
            pred_query,
            query_batch,
            eta=adapted_eta,
            eta_ref=eta0_task,
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
        }

    def _evaluate_batches(self, support_batch, query_batch):
        self.meta_net.eval()
        adapted_eta, eta0_task = self._adapt_eta(support_batch)

        with torch.no_grad():
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
            total_loss, recon_loss, param_loss, eta_reg = self._compute_total_loss(
                self.meta_net,
                pred,
                query_batch,
                eta=adapted_eta,
                eta_ref=eta0_task,
                include_eta_reg=True,
                aux=query_aux,
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

        self.meta_net.train()
        return {
            "loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "param_loss": float(param_loss.item()),
            "eta_reg": float(eta_reg.item()),
            "bloch_mse": float(bloch_mse.item()),
            "mse_x": float(mse_x.item()),
            "mse_y": float(mse_y.item()),
            "mse_z": float(mse_z.item()),
            "mse_delta": float(mse_delta.item()),
            "mse_gamma": float(mse_gamma.item()),
            "pred_A": float(query_aux["pred_A"].item()),
            "pred_alpha": float(query_aux["pred_alpha"].item()),
            "pred_r": float(query_aux["pred_r"].item()),
            "err_alpha": float(err_alpha.item()),
            "err_r": float(err_r.item()),
            "pred": pred,
            "pred_delta": pred_delta,
            "pred_gamma": pred_gamma,
            "batch": query_batch,
            "eta": adapted_eta.detach().cpu().numpy(),
            "eta0_task": eta0_task.detach().cpu().numpy(),
        }

    def train(self, num_epochs=200, tasks_per_epoch=5, val_tasks=None):
        ids = sorted(self.task_dataset.keys())
        for ep in range(num_epochs):
            chosen = ids
            ep_loss = 0.0
            ep_recon = 0.0
            ep_param = 0.0
            all_eta_shift = []
            all_eta_norm = []
            for tid in chosen:
                update_info = self.meta_update(self.task_dataset[tid])
                ep_loss += update_info["loss"]
                ep_recon += update_info["recon_loss"]
                ep_param += update_info["param_loss"]
                all_eta_shift.append(update_info["eta_shift"])
                all_eta_norm.append(float(np.linalg.norm(update_info["adapted_eta"])))
                print(
                    f"Ep {ep:3d} T{tid}: query_loss={update_info['loss']:.6f} "
                    f"recon={update_info['recon_loss']:.6f} "
                    f"param={update_info['param_loss']:.6f} "
                    f"eta_reg={update_info['eta_reg']:.6f} "
                    f"|adapted_eta|={np.linalg.norm(update_info['adapted_eta']):.4f} "
                    f"|meta_eta|={np.linalg.norm(update_info['meta_eta']):.4f} "
                    f"eta_shift={update_info['eta_shift']:.5f}"
                )

            avg = ep_loss / len(chosen)
            avg_recon = ep_recon / len(chosen)
            avg_param = ep_param / len(chosen)
            self.metrics_his.append(avg)
            self.train_recon_his.append(avg_recon)
            self.train_param_his.append(avg_param)
            self.scheduler.step()

            val_avg = None
            selection_score = None
            if val_tasks is not None:
                val_loss = 0.0
                val_recon = 0.0
                val_param = 0.0
                for _, task_data in val_tasks.items():
                    self.meta_net.load_state_dict(self.meta_net.state_dict())
                    support_batch, query_batch = self._deterministic_support_query(task_data)
                    metrics = self._evaluate_batches(support_batch, query_batch)
                    val_loss += metrics["loss"]
                    val_recon += metrics["recon_loss"]
                    val_param += metrics["param_loss"]

                val_avg = val_loss / len(val_tasks)
                val_recon_avg = val_recon / len(val_tasks)
                val_param_avg = val_param / len(val_tasks)
                # Select checkpoints by parameter identification first, with reconstruction as a tie-breaker.
                selection_score = val_param_avg + 0.1 * val_recon_avg
                self.val_metrics_his.append(val_avg)
                self.val_recon_his.append(val_recon_avg)
                self.val_param_his.append(val_param_avg)
                self.selection_score_his.append(selection_score)
                if selection_score < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.lowest_loss = selection_score
            else:
                if avg < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.lowest_loss = avg

            val_str = f"  val={val_avg:.6f}" if val_avg is not None else ""
            select_str = f"  sel={selection_score:.6f}" if selection_score is not None else ""
            eta_shift_avg = float(np.mean(all_eta_shift)) if all_eta_shift else 0.0
            eta_norm_avg = float(np.mean(all_eta_norm)) if all_eta_norm else float(torch.norm(self.meta_net.eta_init).item())
            self.train_eta_shift_his.append(eta_shift_avg)
            self.train_eta_norm_his.append(eta_norm_avg)
            print(
                f"  Avg query loss: {avg:.6f}{val_str}{select_str}  "
                f"mean_eta_norm={eta_norm_avg:.4f}  "
                f"mean_eta_shift={eta_shift_avg:.5f}"
            )

    def evaluate(self, task_data):
        self.meta_net.load_state_dict(self.best_net.state_dict())
        support_batch, query_batch = self._deterministic_support_query(task_data)
        return self._evaluate_batches(support_batch, query_batch)

    def predict_full_trajectory(self, task_data, traj_idx=0):
        self.meta_net.load_state_dict(self.best_net.state_dict())

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
        pred_full = metrics["pred"][:, 0, :].detach().cpu().numpy()
        true_traj = query_batch["bloch"][:, 0, :3].detach().cpu().numpy()
        true_delta = query_batch["delta"][:, 0].detach().cpu().numpy()
        true_gamma = query_batch["gamma"][:, 0].detach().cpu().numpy()
        pred_traj = pred_full.copy()
        pred_delta = metrics["pred_delta"][:, 0].detach().cpu().numpy()
        pred_gamma = metrics["pred_gamma"][:, 0].detach().cpu().numpy()

        return {
            "loss": metrics["loss"],
            "bloch_mse": metrics["bloch_mse"],
            "mse_x": metrics["mse_x"],
            "mse_y": metrics["mse_y"],
            "mse_z": metrics["mse_z"],
            "mse_delta": metrics["mse_delta"],
            "mse_gamma": metrics["mse_gamma"],
            "param_loss": metrics["param_loss"],
            "pred_A": metrics["pred_A"],
            "pred_alpha": metrics["pred_alpha"],
            "pred_r": metrics["pred_r"],
            "err_alpha": metrics["err_alpha"],
            "err_r": metrics["err_r"],
            "eta": metrics["eta"],
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
                "mse_delta": metrics["mse_delta"],
                "mse_gamma": metrics["mse_gamma"],
                "param_loss": metrics["param_loss"],
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
            }
        return results
