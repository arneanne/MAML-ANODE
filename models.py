# models.py - Meta-AQNODE: MAML architecture
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint as odeint_orig


def interpolate_dY_batch(t, t_grid, dY_seq):
    if dY_seq.numel() == 0:
        return torch.zeros(dY_seq.shape[0], device=dY_seq.device)
    if t <= t_grid[0]:
        return dY_seq[:, 0]
    if t >= t_grid[-1]:
        return dY_seq[:, -1]

    idx = torch.searchsorted(t_grid, t.detach().to(t_grid.device)).item() - 1
    idx = max(0, min(int(idx), dY_seq.shape[1] - 2))
    t0, t1 = t_grid[idx], t_grid[idx + 1]
    coef = (t - t0) / (t1 - t0)
    coef = coef.to(dY_seq.device)
    return dY_seq[:, idx] * (1 - coef) + dY_seq[:, idx + 1] * coef


class Encoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, t, h, dY, lam):
        while dY.dim() < h.dim():
            dY = dY.unsqueeze(-1)
        if dY.shape[0] != h.shape[0]:
            dY = dY.expand(h.shape[0], -1)
        lam_exp = lam.unsqueeze(0).expand(h.shape[0], -1)
        return self.net(torch.cat([h, dY, lam_exp], dim=-1))


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 5),
        )

    def forward(self, h):
        return self.net(h)


class AQNodeBase(nn.Module):
    """Base AQNode shared across tasks: encoder + ODE func + decoder."""

    def __init__(self, latent_dim=64, measurement_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.measurement_dim = measurement_dim
        self.encoder = Encoder(3 + 2 + measurement_dim, 64, latent_dim)
        self.ode_func = LatentODEFunc(latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode_initial(self, init_state, dY_seq, init_delta, init_gamma):
        """Return augmented h0 = [latent_state | truncated_dY_seq]."""
        batch_size = init_state.shape[0]
        width = min(dY_seq.shape[-1], self.measurement_dim)
        dY_feat = dY_seq[:, :width]
        if width < self.measurement_dim:
            pad = torch.zeros(
                batch_size,
                self.measurement_dim - width,
                device=dY_seq.device,
            )
            dY_feat = torch.cat([dY_feat, pad], dim=-1)

        enc_in = torch.cat([init_state, init_delta, init_gamma, dY_feat], dim=-1)
        latent_state = self.encoder(enc_in)
        return torch.cat([latent_state, dY_feat], dim=-1)

    def forward(self, h0, t_grid, lam, return_h=False, method="dopri5"):
        t_grid = t_grid.to(h0.device)

        def ode_func(t, h):
            latent_state = h[:, : self.latent_dim]
            meas_seq = h[:, self.latent_dim : self.latent_dim + self.measurement_dim]
            dY_t = interpolate_dY_batch(t, t_grid, meas_seq).unsqueeze(-1)
            latent_deriv = self.ode_func(t, latent_state, dY_t, lam)
            return torch.cat([latent_deriv, torch.zeros_like(meas_seq)], dim=-1)

        h_traj = odeint_orig(ode_func, h0, t_grid, method=method, atol=1e-6, rtol=1e-6)
        pred = self.decoder(h_traj[:, :, : self.latent_dim])
        return (pred, h_traj) if return_h else pred


class InnerAQNode(nn.Module):
    """AQNode with inner-loop learnable lambda parameters (alpha, r)."""

    def __init__(self, latent_dim=64, lambda_dim=2, measurement_dim=50):
        super().__init__()
        self.base = AQNodeBase(latent_dim, measurement_dim)
        self.lambda_param = nn.Parameter(torch.tensor([0.5, 0.3]))
        self.latent_dim = latent_dim
        self.lambda_dim = lambda_dim

    def clone(self):
        clone = InnerAQNode(self.latent_dim, self.lambda_dim, self.base.measurement_dim)
        clone.base.load_state_dict(self.base.state_dict())
        clone.lambda_param.data.copy_(self.lambda_param.data)
        return clone

    def encode_initial(self, init_state, dY_seq, init_delta, init_gamma):
        return self.base.encode_initial(init_state, dY_seq, init_delta, init_gamma)

    def forward(self, h0, t_grid, lam=None, return_h=False, method="dopri5"):
        if lam is None:
            lam = self.lambda_param
        return self.base.forward(h0, t_grid, lam, return_h, method)

    def compute_loss(
        self,
        pred,
        bloch_target,
        delta_target,
        gamma_target,
        w_z=1.5,
        w_delta=0.7,
        w_gamma=1.2,
        init_weight=0.1,
        bloch_reg=0.0,
        trend_weight=0.0,
    ):
        """Weighted component loss aligned with train/eval settings."""
        loss_x = torch.mean((pred[1:, :, 0] - bloch_target[1:, :, 0]) ** 2)
        loss_y = torch.mean((pred[1:, :, 1] - bloch_target[1:, :, 1]) ** 2)
        loss_z = torch.mean((pred[1:, :, 2] - bloch_target[1:, :, 2]) ** 2)
        loss_d = torch.mean((pred[1:, :, 3] - delta_target[1:]) ** 2)
        loss_g = torch.mean((pred[1:, :, 4] - gamma_target[1:]) ** 2)

        loss_init = init_weight * torch.mean(
            (pred[0:1, :, :3] - bloch_target[0:1, :, :3]) ** 2
        )

        loss_reg = 0.0
        if bloch_reg > 0:
            radius = torch.linalg.norm(pred[:, :, :3], dim=-1)
            loss_reg = bloch_reg * torch.mean(torch.relu(radius - 1.0) ** 2)

        loss_trend = 0.0
        if trend_weight > 0 and pred.shape[0] > 1:
            pred_diff = pred[1:, :, :3] - pred[:-1, :, :3]
            target_diff = bloch_target[1:, :, :3] - bloch_target[:-1, :, :3]
            loss_trend = trend_weight * torch.mean((pred_diff - target_diff) ** 2)

        return (
            loss_x
            + loss_y
            + w_z * loss_z
            + w_delta * loss_d
            + w_gamma * loss_g
            + loss_init
            + loss_reg
            + loss_trend
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
        perturb_scale=0.05,
        w_z=1.5,
        w_gamma=1.2,
        bloch_reg=0.01,
        inner_steps=5,
        inner_lr=0.1,
        lambda_meta_lr=0.2,
        lambda_supervision_weight=1.0,
        trend_weight=0.5,
        device="cuda:0",
        init_weight=0.1,
        measurement_dim=None,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.task_dataset = task_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.perturb_scale = perturb_scale
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.lambda_meta_lr = lambda_meta_lr
        self.lambda_supervision_weight = lambda_supervision_weight
        self.trend_weight = trend_weight
        self.init_weight = init_weight
        self.w_z = w_z
        self.w_delta = 0.7
        self.w_gamma = w_gamma
        self.bloch_reg = bloch_reg
        self.measurement_dim = measurement_dim if measurement_dim is not None else seq_len

        self.meta_net = InnerAQNode(latent_dim, measurement_dim=self.measurement_dim).to(self.device)
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

    def _compute_recon_loss(self, model, pred, batch):
        return model.compute_loss(
            pred,
            batch["bloch"],
            batch["delta"],
            batch["gamma"],
            w_z=self.w_z,
            w_delta=self.w_delta,
            w_gamma=self.w_gamma,
            init_weight=self.init_weight,
            bloch_reg=self.bloch_reg,
            trend_weight=self.trend_weight,
        )

    def _compute_lambda_loss(self, lam, batch):
        target_lambda = batch["task_lambda"]
        return torch.mean((lam - target_lambda) ** 2)

    def _compute_total_loss(self, model, pred, batch, lam=None, include_lambda_supervision=False):
        recon_loss = self._compute_recon_loss(model, pred, batch)
        lambda_loss = torch.tensor(0.0, device=self.device)
        if include_lambda_supervision and lam is not None and self.lambda_supervision_weight > 0:
            lambda_loss = self.lambda_supervision_weight * self._compute_lambda_loss(lam, batch)
        total_loss = recon_loss + lambda_loss
        return total_loss, recon_loss.detach(), lambda_loss.detach()

    def _build_batch(self, task_data, traj_idx, t_start=0, seq_len=None, apply_perturb=True):
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

        actual_batch = len(traj_idx)
        init_state = bloch[0].clone()
        width = min(self.measurement_dim, dY.shape[0])
        dY_seq_flat = dY[:width].permute(1, 0, 2).reshape(actual_batch, -1)
        if width < self.measurement_dim:
            pad = torch.zeros(actual_batch, self.measurement_dim - width, device=dY.device)
            dY_seq_flat = torch.cat([dY_seq_flat, pad], dim=-1)

        init_delta = delta[0].clone().unsqueeze(-1)
        init_gamma = gamma[0].clone().unsqueeze(-1)
        if apply_perturb and self.perturb_scale > 0:
            init_state += torch.randn_like(init_state) * self.perturb_scale
            init_delta += torch.randn_like(init_delta) * self.perturb_scale * 0.3
            init_gamma += torch.randn_like(init_gamma) * self.perturb_scale * 0.3

        return {
            "t_grid": t_grid,
            "dY": dY,
            "bloch": bloch,
            "delta": delta,
            "gamma": gamma,
            "task_lambda": torch.tensor(
                [task_data["alpha"], task_data["r"]],
                dtype=torch.float32,
                device=self.device,
            ),
            "init_state": init_state,
            "dY_seq_flat": dY_seq_flat,
            "init_delta": init_delta,
            "init_gamma": init_gamma,
            "traj_idx": traj_idx,
            "t_start": t_start,
        }

    def sample_task_data(self, task_data, seq_len=None, apply_perturb=True):
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
            apply_perturb=apply_perturb,
        )

    def _sample_support_query(self, task_data):
        support = self.sample_task_data(task_data, apply_perturb=True)
        query = self.sample_task_data(task_data, apply_perturb=True)
        for _ in range(3):
            same_traj = np.array_equal(support["traj_idx"], query["traj_idx"])
            same_window = support["t_start"] == query["t_start"]
            if not (same_traj and same_window):
                break
            query = self.sample_task_data(task_data, apply_perturb=True)
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
            apply_perturb=False,
        )
        query = self._build_batch(
            task_data,
            traj_idx=query_idx,
            t_start=0,
            seq_len=seq_len,
            apply_perturb=False,
        )
        return support, query

    def _adapt_lambda(self, support_batch):
        h0 = self.meta_net.encode_initial(
            support_batch["init_state"],
            support_batch["dY_seq_flat"],
            support_batch["init_delta"],
            support_batch["init_gamma"],
        )
        h0_inner = h0.detach()

        adapted_lambda = nn.Parameter(self.meta_net.lambda_param.detach().clone().to(self.device))
        inner_optim = optim.SGD([adapted_lambda], lr=self.inner_lr)

        for _ in range(self.inner_steps):
            pred = self.meta_net(h0_inner, support_batch["t_grid"], lam=adapted_lambda)
            loss, _, _ = self._compute_total_loss(
                self.meta_net,
                pred,
                support_batch,
                lam=adapted_lambda,
                include_lambda_supervision=True,
            )
            inner_optim.zero_grad()
            loss.backward()
            inner_optim.step()

        return adapted_lambda.detach()

    def _clear_inner_loop_grads(self):
        self.meta_net.base.encoder.zero_grad()
        self.meta_net.base.ode_func.zero_grad()
        self.meta_net.base.decoder.zero_grad()
        self.meta_net.lambda_param.grad = None

    def _meta_update_lambda_init(self, adapted_lambda):
        with torch.no_grad():
            self.meta_net.lambda_param.data.lerp_(
                adapted_lambda.to(self.meta_net.lambda_param.device),
                self.lambda_meta_lr,
            )

    def meta_update(self, task_data):
        """Single FOMAML update with separate support/query batches."""
        support_batch, query_batch = self._sample_support_query(task_data)
        self.meta_net.train()
        self.optimizer.zero_grad()

        lambda_before = self.meta_net.lambda_param.detach().clone()
        adapted_lambda = self._adapt_lambda(support_batch)
        self._clear_inner_loop_grads()

        h0_query = self.meta_net.encode_initial(
            query_batch["init_state"],
            query_batch["dY_seq_flat"],
            query_batch["init_delta"],
            query_batch["init_gamma"],
        )
        pred_query = self.meta_net(h0_query, query_batch["t_grid"], lam=adapted_lambda)
        meta_loss, recon_loss, lambda_loss = self._compute_total_loss(
            self.meta_net,
            pred_query,
            query_batch,
            lam=adapted_lambda,
            include_lambda_supervision=True,
        )

        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.meta_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        self._meta_update_lambda_init(adapted_lambda)

        lambda_after = self.meta_net.lambda_param.detach().clone().cpu().numpy()
        lam_shift = float(torch.norm(adapted_lambda.cpu() - lambda_before.cpu()).item())

        return {
            "loss": meta_loss.item(),
            "recon_loss": float(recon_loss.item()),
            "lambda_loss": float(lambda_loss.item()),
            "adapted_lam": adapted_lambda.cpu().numpy(),
            "meta_lam": lambda_after,
            "lam_shift": lam_shift,
        }

    def _evaluate_batches(self, support_batch, query_batch):
        self.meta_net.eval()
        adapted_lambda = self._adapt_lambda(support_batch)
        self._clear_inner_loop_grads()

        with torch.no_grad():
            h0_query = self.meta_net.encode_initial(
                query_batch["init_state"],
                query_batch["dY_seq_flat"],
                query_batch["init_delta"],
                query_batch["init_gamma"],
            )
            pred = self.meta_net(h0_query, query_batch["t_grid"], lam=adapted_lambda)
            total_loss, recon_loss, lambda_loss = self._compute_total_loss(
                self.meta_net,
                pred,
                query_batch,
                lam=adapted_lambda,
                include_lambda_supervision=True,
            )
            bloch_mse = torch.mean((pred[:, :, :3] - query_batch["bloch"]) ** 2)

        self.meta_net.train()
        return {
            "loss": float(total_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "lambda_loss": float(lambda_loss.item()),
            "bloch_mse": float(bloch_mse.item()),
            "pred": pred,
            "batch": query_batch,
            "lam": adapted_lambda.cpu().numpy(),
        }

    def train(self, num_epochs=200, tasks_per_epoch=5, val_tasks=None):
        ids = list(self.task_dataset.keys())
        for ep in range(num_epochs):
            chosen = np.random.choice(ids, min(tasks_per_epoch, len(ids)), replace=False)
            ep_loss = 0.0
            all_adapted_lam = []
            all_meta_lam = []
            all_lam_shift = []
            for tid in chosen:
                update_info = self.meta_update(self.task_dataset[tid])
                ep_loss += update_info["loss"]
                all_adapted_lam.append(update_info["adapted_lam"])
                all_meta_lam.append(update_info["meta_lam"])
                all_lam_shift.append(update_info["lam_shift"])
                print(
                    f"Ep {ep:3d} T{tid}: query_loss={update_info['loss']:.6f} "
                    f"recon={update_info['recon_loss']:.6f} "
                    f"lam_loss={update_info['lambda_loss']:.6f} "
                    f"adapted_lam=[{update_info['adapted_lam'][0]:.4f} {update_info['adapted_lam'][1]:.4f}] "
                    f"meta_lam=[{update_info['meta_lam'][0]:.4f} {update_info['meta_lam'][1]:.4f}] "
                    f"lam_shift={update_info['lam_shift']:.5f}"
                )

            avg = ep_loss / len(chosen)
            self.metrics_his.append(avg)
            self.scheduler.step()

            val_avg = None
            if val_tasks is not None:
                val_loss = 0.0
                for _, task_data in val_tasks.items():
                    self.meta_net.load_state_dict(self.meta_net.state_dict())
                    support_batch, query_batch = self._deterministic_support_query(task_data)
                    metrics = self._evaluate_batches(support_batch, query_batch)
                    val_loss += metrics["loss"]

                val_avg = val_loss / len(val_tasks)
                self.val_metrics_his.append(val_avg)
                if val_avg < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.lowest_loss = val_avg
            else:
                if avg < self.lowest_loss:
                    self.best_net = copy.deepcopy(self.meta_net)
                    self.lowest_loss = avg

            val_str = f"  val={val_avg:.6f}" if val_avg is not None else ""
            lam_shift_avg = float(np.mean(all_lam_shift)) if all_lam_shift else 0.0
            meta_lam_mean = np.mean(all_meta_lam, axis=0) if all_meta_lam else self.meta_net.lambda_param.detach().cpu().numpy()
            print(
                f"  Avg query loss: {avg:.6f}{val_str}  "
                f"mean_meta_lam=[{meta_lam_mean[0]:.4f} {meta_lam_mean[1]:.4f}]  "
                f"mean_lam_shift={lam_shift_avg:.5f}"
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
            apply_perturb=False,
        )
        query_batch = self._build_batch(
            task_data,
            traj_idx=np.asarray([traj_idx]),
            t_start=0,
            seq_len=task_data["bloch"].shape[1],
            apply_perturb=False,
        )
        metrics = self._evaluate_batches(support_batch, query_batch)
        pred_full = metrics["pred"][:, 0, :].detach().cpu().numpy()
        true_traj = query_batch["bloch"][:, 0, :3].detach().cpu().numpy()
        true_delta = query_batch["delta"][:, 0].detach().cpu().numpy()
        true_gamma = query_batch["gamma"][:, 0].detach().cpu().numpy()
        pred_traj = pred_full[:, :3].copy()
        pred_traj[0] = true_traj[0]
        pred_delta = pred_full[:, 3].copy()
        pred_gamma = pred_full[:, 4].copy()
        pred_delta[0] = true_delta[0]
        pred_gamma[0] = true_gamma[0]

        return {
            "loss": metrics["loss"],
            "bloch_mse": metrics["bloch_mse"],
            "lam": metrics["lam"],
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
                "alpha": task_data["alpha"],
                "r": task_data["r"],
                "lam": metrics["lam"],
            }
        return results
