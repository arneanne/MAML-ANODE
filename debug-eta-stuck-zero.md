# Debug Session: eta-stuck-zero
- **Status**: [OPEN]
- **Issue**: `eta` remains all zeros during training and evaluation, so task adaptation appears inactive.
- **Debug Server**: not-started
- **Log File**: .dbg/trae-debug-log-eta-stuck-zero.ndjson

## Reproduction Steps
1. Run `python train.py` with any configuration that reaches training and evaluation.
2. Inspect logged `adapted_eta`, `meta_eta`, and saved `eta` fields in `results/*/test_results.json`.
3. Observe that all `eta` entries remain zero vectors.

## Hypotheses & Verification
| ID | Hypothesis | Likelihood | Effort | Evidence |
|----|------------|------------|--------|----------|
| A | `eta` receives near-zero gradients because the model uses `alpha/r` to explain the task and barely depends on `eta`. | High | Low | Confirmed |
| B | `adapted_eta.grad` exists in the inner loop but is not transferred into `eta_init.grad` correctly in the outer loop. | High | Medium | Rejected |
| C | A detach / clone boundary breaks the computation graph between the loss and `eta`. | Medium | Medium | Rejected |
| D | `eta` is updated, but later code overwrites or serializes the pre-update zero value. | Low | Low | Rejected |

## Log Evidence
- `.dbg/trae-debug-log-eta-stuck-zero.ndjson:2` and `:4` show `eta_grad_norm = 0.0` in the inner loop, while `task_alpha_grad` and `task_r_grad` are non-zero.
- `.dbg/trae-debug-log-eta-stuck-zero.ndjson:6` and `:7` show `adapted_eta_grad_norm = 0.0` and transferred `eta_init_grad_norm_after_transfer = 0.0`.
- `.dbg/trae-debug-log-eta-stuck-zero.ndjson:8` shows `eta_init_delta_norm = 0.0` after the outer optimizer step.
- Static inspection of `models.py` shows that when `task_params_override` is provided, `forward()` skips `_infer_task_params(h_traj, eta)` and uses only overridden `pred_alpha/pred_r`; `pred_bloch` is then computed solely from `init_state` and formula-derived `Delta/gamma`.
- Post-fix verification in `.dbg/trae-debug-log-eta-stuck-zero.ndjson:3-9` shows `eta_grad_norm = 7.98e-05` in the inner loop, `adapted_eta_grad_norm = 1.15e-02` in the outer loop, and `eta_init_delta_norm = 3.9999e-02` after the optimizer step.
- Post-fix evaluation output at `results/debug_eta_fix_probe/test_results.json` records a non-zero `eta` vector instead of all zeros.

## Verification Conclusion
- Root cause: during both the inner loop and outer meta-update, the loss path uses `task_params_override={"pred_alpha": task_alpha, "pred_r": task_r}`. In this branch, `eta` only affects `h_traj`, but `h_traj` is not used to build `pred_bloch` or the parameter loss. Therefore the loss is mathematically independent of `eta`, making its gradient exactly zero.
- Consequence: `adapted_eta` never updates in `_adapt_eta()`, and the manual transfer `self.meta_net.eta_init.grad = adapted_eta.grad.detach().clone()` only copies zeros in `meta_update()`.
- Applied fix: removed the independent inner-loop `task_alpha/task_r` parameters, made the inner loop optimize only `eta`, and removed query/support `task_params_override` so each forward pass re-infers `alpha,r` from the current `eta`.
- Post-fix result: `eta` now participates in the loss path and updates successfully, but the broader training quality of `alpha/r` still needs further evaluation on a full run.
