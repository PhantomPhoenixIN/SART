# Compute and inference profiling protocol

The architecture comparison reports separate operator-estimated prefill FLOPs, recurrent-module FLOPs, and cached-decode FLOPs, plus an estimated total. `scripts/evaluate/profile_inference_cost.py` profiles a fixed source prefix and replays the same number of reference target tokens from an XLCoST validation example for every architecture. Example commands:

```bash
env/bin/python scripts/evaluate/profile_inference_cost.py \
  --checkpoint checkpoints/confirmatory_7b/<experiment-id>/best.pt \
  --validation-root content/datasets/xlcost/structural_external_clean \
  --sample-index 42 --decode-tokens 128 --hard-threshold 0.5 \
  --output outputs/compute_profiles/<experiment-id>_42.json
```

Use the same validation example indices, source lengths and replay lengths for all compared checkpoints. Aggregate several examples, and report the exact indices. Sweep the predefined validation-only threshold grid (or matched fixed recurrence budgets) and pass each threshold with `--hard-threshold`. The optional fixed-budget/top-k policy uses the frozen `inference_top_k_fraction_validation_grid` from `configs/training/search.yaml`; pass one fraction with `--top-k-fraction`, which is mutually exclusive with the threshold. It retains a stable top fraction of the still-active units independently per example at each recurrent step. Hard inference routing physically gathers active tokens/regions into the shared recurrent core and skips halted units; graph messages to active regions may still read halted neighbors' frozen states. This realizes adaptive recurrent FLOPs instead of treating gate values only as update weights. The profiler records CUDA-event/device timings for prefill, the recurrent adapter, and cached decode, plus peak allocated VRAM. It also records PyTorch operator FLOP estimates and a separately labeled dense-attention upper estimate for fused SDPA work, which PyTorch's profiler may not count. `estimated_recurrent_module_flops` is reported separately and overlaps the prefill operator total; do not add it a second time to `estimated_total_flops`.

The fixed-length profile isolates architectural compute under identical sequence lengths. Natural greedy-generation latency and generated length are reported separately from per-task external evaluation records. Compute-budget calibration and hard-halting threshold selection use only in-domain validation. External evaluation receives a frozen configuration and threshold. Report both profiler estimates and measured wall/device time, because the attention estimate and FLOP profiler are approximations rather than hardware-independent exact counts.

## Confirmatory profile dispatch

`configs/evaluation/efficiency_profile_sample.json` freezes the 24-example, six-direction in-domain validation sample used for the final profile set. `scripts/evaluate/dispatch_efficiency_profiles.py --cap 4 --weekend-cap 4 --interval-seconds 60` waits until all55 canonical runs are trained, the external protocol is frozen, and all frozen external prediction outputs are complete. It then schedules one profile per canonical best checkpoint, passes that checkpoint's frozen routing threshold/top-k policy, and writes verified records under `reports/profiling/<backbone-revision>/<experiment-id>.json`. It shares the GPU scheduler lock and counts all GPU queue jobs toward the same four-job cap. External model inference has priority: profiling does not begin until the prediction dispatcher has completed all benchmark tasks.
