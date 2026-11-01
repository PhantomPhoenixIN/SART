# Baseline fidelity

This table separates source-level implementation fidelity from results. The shared experiment grid has not started; the only current 7B jobs are validation-only Dense/SART convergence pilots.

| Baseline | Implementation correspondence | Fidelity label and deviations |
|---|---|---|
| Dense | Same-backbone causal SFT with shared source/target serialization and target-only loss. | Dense baseline. |
| Dense-ParamMatch | Non-recurrent residual MLP adapter sized to match Full-SART added trainable parameters within 0.012% at the Qwen hidden size. | Capacity control; does not use structure, routing or recurrence. |
| Uniform Token Loop | Shared recurrent token transformation over mapped source-code tokens, fixed configured depth, content-only self-attention. | Looped-Transformer baseline. |
| Uniform Region Loop | Same recurrent core over semantic regions, every region gets the full fixed depth. | Uniform semantic-region recurrence. |
| MoR | Token-level dynamic recurrence over source-code tokens, a content-only recurrent self-attention block, current-state sigmoid router, hard threshold or optional fixed-fraction top-k routing, and active-token skipping at inference. | **MoR-style token-adaptive recurrence**, not an exact reproduction. It uses one compact inserted recurrent module rather than the published full shared stack and does not reproduce published KV sharing. Compatible Q/K/V/output weights are initialized from the same recurrent module construction used by SART. |
| Random-region adaptive | Deterministic random contiguous regions with semantic region-size distribution; adaptive recurrent routing without real graph relations. | Random-region control. |
| RegionAdaptive-NoGraph | Semantic region boundaries and pooled content; no region labels, graph edges, structural scalars or parent relations enter the router/core. | Isolates region grouping/content from explicit structure. |
| Shuffled-Structure SART | Permuted typed edges and region/scalar labels, retaining node/edge counts and group lengths. | Structure-destruction control. |
| Structure-only / AST/CFG/DFG SART | Graph/region representations with one pass or selected edge families. | Structural and graph-family ablations, not new architectures. |
| Full SART | Language-normalized semantic regions, typed AST/CFG/DFG messages, source-only adaptive region recurrence and hard sparse inference routing. | Proposed method; no confirmatory result yet. |

## Shared comparison controls

- Every recurrent method runs at the same mid-backbone insertion fraction, width and source-prefix positions.
- The token and region methods instantiate the same recurrent core modules with the same run seed, including shared Q/K/V/output initialization for self-attention/graph messages. Mode-specific routing uses the matching token or region router; no target teacher-forced tokens enter recurrent computation.
- The six-direction prompt and output extraction are frozen in `docs/TRANSLATION_PROTOCOL.md`.
- All methods use the same clean XLCoST pair manifest, balanced direction sampler, fixed epoch horizon and optimizer-step/token budget. Checkpoints are selected on in-domain validation only.
- Published systems with different backbones/data/decoding remain contextual references; their scores do not enter the controlled same-backbone table.

## Literature fidelity risks

- Universal Transformer/ACT established shared recurrent computation and per-position halting.
- StructCoder established AST/DFG-aware code representations and target structural objectives.
- Mixture-of-Recursions (MoR) established shared recursive computation with token-level dynamic depth. Its official implementation is Llama-oriented and documents a newer CUDA/PyTorch stack than this private cu118 environment. The controlled adapter is therefore labeled MoR-style; the deviations above must remain visible in papers/tables.
- LoopCoder (Findings ACL 2026) is a recent scaled looped code model. Do not use LoopCoder as the name of the local fixed-depth recurrent control.
- bioMoR (arXiv:2608.06727) is close conceptual prior art for graph-conditioned recurrent allocation in another domain. Do not claim graph-aware adaptive recurrence as a general first.

## Status history

- 2026-10-02: corrected MoR-style router inputs to use current recurrent token states and added gradient-flow coverage.
- 2026-10-02: implemented hard token/region inference by gathering active units into recurrent core calls; halted units are skipped. For graph SART, only active receivers form messages and stopped region representations remain available as neighbors. The `structure_only` control executes exactly one recurrent pass. Added call-count tests. No MoR training or external predictions have run.
