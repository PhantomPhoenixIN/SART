# Initial xCodeEval Common-Core results — seed 42

**Preliminary accuracy preview.** One 7B seed across 332 problems and six translation directions. Pass@1 uses one seeded stochastic candidate per task under frozen Protocol v3. These results cannot confirm the matched-compute hypothesis; the three-seed analysis and inference profiles are still required.

| Architecture | Pass@1 macro | Runner/setup errors |
|---|---:|---:|
| full_sart | 0.2595 | 0 |
| dense | 0.2641 | 0 |
| mor_style | 0.2620 | 0 |
| uniform_token_loop | 0.2912 | 0 |

| Full-SART contrast | Difference (percentage points) | Exploratory 95% problem bootstrap interval |
|---|---:|---:|
| full_sart_vs_dense | -0.45 | [-2.56, +1.61] |
| full_sart_vs_mor_style | -0.25 | [-2.26, +1.81] |
| full_sart_vs_uniform_token_loop | -3.16 | [-5.27, -1.00] |

Single seed and no matched end-to-end compute profile yet; this preview cannot confirm H1 or H2. Three-seed paired inference, all planned controls, and efficiency profiles remain required.

Full per-direction rates, score manifest hashes, and experiment IDs are in the companion JSON.
