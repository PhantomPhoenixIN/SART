# In-domain validation screen

Generated from the canonical registry and per-run `metrics.jsonl` histories. For each trained Qwen2.5-Coder-7B confirmatory run, the displayed epoch is selected by minimum XLCoST validation target loss, following the frozen checkpoint-selection rule.

**Scope:** descriptive XLCoST validation only. These values are not external benchmark results, do not establish functional correctness, and are not inferential comparisons. Runs with incomplete seed replication are shown individually and must not be treated as balanced architecture means. The report does not read external manifests or predictions.

Canonical trained runs included: **40**. Each selected epoch is within the fixed two-epoch horizon.

## Architecture-level descriptive means

Means use only currently TRAINED seeds; the `n/seeds` column makes incomplete replication visible. Do not compare these partial means as confirmatory effects.

| Architecture | n / seeds | Best epoch mean | Val loss | BLEU-4 | CodeBLEU | CrystalBLEU | Exact match | Mean depth | Stop at 1 | Reach Rmax | Router entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ast_cfg_sart | 1 / 42 | 2.00 | 0.0682 | 0.8737 | 0.8472 | 0.8637 | 0.0417 | 3.0000 | 0.0150 | 0.0003 | 0.2003 |
| ast_dfg_sart | 1 / 42 | 2.00 | 0.0682 | 0.8729 | 0.8483 | 0.8641 | 0.0417 | 3.0000 | 0.0171 | 0.0005 | 0.2279 |
| ast_sart | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8697 | 0.8388 | 0.8517 | 0.0556 | 2.9999 | 0.0122 | 0.0002 | 0.1630 |
| cfg_sart | 1 / 42 | 2.00 | 0.0682 | 0.8745 | 0.8479 | 0.8653 | 0.0417 | 3.0000 | 0.0309 | 0.0033 | 0.3072 |
| dense | 3 / 13,42,87 | 2.00 | 0.0684 | 0.8703 | 0.8377 | 0.8536 | 0.0625 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dense_param_match | 3 / 13,42,87 | 2.00 | 0.0681 | 0.8707 | 0.8375 | 0.8550 | 0.0590 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| dfg_sart | 1 / 42 | 2.00 | 0.0682 | 0.8758 | 0.8491 | 0.8655 | 0.0417 | 2.9998 | 0.0327 | 0.0035 | 0.3176 |
| full_sart | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8690 | 0.8380 | 0.8522 | 0.0556 | 2.9999 | 0.0116 | 0.0002 | 0.1614 |
| mor_style | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8695 | 0.8389 | 0.8519 | 0.0556 | 3.0000 | 0.2103 | 0.1132 | 0.6378 |
| random_region | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8701 | 0.8395 | 0.8533 | 0.0556 | 3.0001 | 0.0348 | 0.0053 | 0.3910 |
| region_adaptive_no_graph | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8700 | 0.8393 | 0.8515 | 0.0556 | 2.9999 | 0.0401 | 0.0037 | 0.3435 |
| sart_attention_pool | 1 / 42 | 2.00 | 0.0682 | 0.8756 | 0.8500 | 0.8647 | 0.0521 | 3.0001 | 0.0115 | 0.0002 | 0.1697 |
| sart_coarse_ontology | 1 / 42 | 2.00 | 0.0682 | 0.8767 | 0.8504 | 0.8681 | 0.0521 | 2.9999 | 0.0191 | 0.0008 | 0.2557 |
| sart_no_language_pair | 1 / 42 | 2.00 | 0.0683 | 0.8776 | 0.8529 | 0.8683 | 0.0417 | 2.9997 | 0.0250 | 0.0011 | 0.2657 |
| sart_no_message | 1 / 42 | 2.00 | 0.0682 | 0.8742 | 0.8487 | 0.8634 | 0.0417 | 2.9998 | 0.0569 | 0.0118 | 0.4718 |
| sart_no_scalar | 1 / 42 | 2.00 | 0.0682 | 0.8731 | 0.8481 | 0.8640 | 0.0417 | 3.0001 | 0.0157 | 0.0004 | 0.2125 |
| shuffled_structure | 1 / 42 | 2.00 | 0.0682 | 0.8742 | 0.8473 | 0.8639 | 0.0312 | 3.0000 | 0.0216 | 0.0010 | 0.2707 |
| structure_only | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8707 | 0.8400 | 0.8535 | 0.0590 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| uniform_region_loop | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8711 | 0.8393 | 0.8554 | 0.0556 | 6.0000 | 0.0000 | 1.0000 | 0.0000 |
| uniform_token_loop | 3 / 13,42,87 | 2.00 | 0.0682 | 0.8710 | 0.8400 | 0.8546 | 0.0521 | 6.0000 | 0.0000 | 1.0000 | 0.0000 |

## Per-run selected validation rows

| Experiment | Seed | Best epoch | Val loss | BLEU-4 | CodeBLEU | CrystalBLEU | Exact match | Mean depth |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ast_cfg_sart-seed42-7015e3075c30` | 42 | 2 | 0.0682 | 0.8737 | 0.8472 | 0.8637 | 0.0417 | 3.0000 |
| `ast_dfg_sart-seed42-e13a33aa0e05` | 42 | 2 | 0.0682 | 0.8729 | 0.8483 | 0.8641 | 0.0417 | 3.0000 |
| `ast_sart-seed13-5116891ac07d` | 13 | 2 | 0.0682 | 0.8516 | 0.8155 | 0.8232 | 0.0729 | 2.9999 |
| `ast_sart-seed42-e618e36b12f2` | 42 | 2 | 0.0682 | 0.8778 | 0.8517 | 0.8692 | 0.0417 | 3.0000 |
| `ast_sart-seed87-76e8fc103817` | 87 | 2 | 0.0681 | 0.8798 | 0.8492 | 0.8625 | 0.0521 | 3.0000 |
| `cfg_sart-seed42-328e86d7eae9` | 42 | 2 | 0.0682 | 0.8745 | 0.8479 | 0.8653 | 0.0417 | 3.0000 |
| `dense-seed13-e0f403916f71` | 13 | 2 | 0.0684 | 0.8595 | 0.8210 | 0.8358 | 0.0833 | 0.0000 |
| `dense-seed42-a2b260ba88c6` | 42 | 2 | 0.0684 | 0.8748 | 0.8454 | 0.8645 | 0.0521 | 0.0000 |
| `dense-seed87-2d7b5b414b0a` | 87 | 2 | 0.0683 | 0.8767 | 0.8468 | 0.8605 | 0.0521 | 0.0000 |
| `dense_param_match-seed13-31f8526940a5` | 13 | 2 | 0.0681 | 0.8549 | 0.8157 | 0.8356 | 0.0625 | 0.0000 |
| `dense_param_match-seed42-02f2a24bc51b` | 42 | 2 | 0.0681 | 0.8750 | 0.8468 | 0.8639 | 0.0625 | 0.0000 |
| `dense_param_match-seed87-19a712888fc9` | 87 | 2 | 0.0681 | 0.8821 | 0.8501 | 0.8656 | 0.0521 | 0.0000 |
| `dfg_sart-seed42-e36943a9b468` | 42 | 2 | 0.0682 | 0.8758 | 0.8491 | 0.8655 | 0.0417 | 2.9998 |
| `full_sart-seed13-5bfb80acbc47` | 13 | 2 | 0.0682 | 0.8529 | 0.8158 | 0.8284 | 0.0729 | 2.9998 |
| `full_sart-seed42-7d9cbd6005d3` | 42 | 2 | 0.0682 | 0.8744 | 0.8477 | 0.8656 | 0.0417 | 3.0001 |
| `full_sart-seed87-259a9284c25f` | 87 | 2 | 0.0682 | 0.8797 | 0.8506 | 0.8627 | 0.0521 | 3.0000 |
| `mor_style-seed13-ec7eb8dd6678` | 13 | 2 | 0.0682 | 0.8518 | 0.8148 | 0.8243 | 0.0625 | 3.0001 |
| `mor_style-seed42-4dbef25dab2b` | 42 | 2 | 0.0682 | 0.8758 | 0.8505 | 0.8666 | 0.0521 | 3.0001 |
| `mor_style-seed87-bf32f4b32152` | 87 | 2 | 0.0682 | 0.8810 | 0.8514 | 0.8648 | 0.0521 | 2.9998 |
| `random_region-seed13-945bac1933bc` | 13 | 2 | 0.0682 | 0.8578 | 0.8231 | 0.8351 | 0.0729 | 3.0004 |
| `random_region-seed42-7bda40b640e2` | 42 | 2 | 0.0682 | 0.8735 | 0.8476 | 0.8638 | 0.0417 | 2.9999 |
| `random_region-seed87-232e4b300d9d` | 87 | 2 | 0.0682 | 0.8789 | 0.8479 | 0.8609 | 0.0521 | 2.9999 |
| `region_adaptive_no_graph-seed13-ca84530ad74b` | 13 | 2 | 0.0682 | 0.8544 | 0.8185 | 0.8247 | 0.0729 | 3.0000 |
| `region_adaptive_no_graph-seed42-23509c481193` | 42 | 2 | 0.0682 | 0.8746 | 0.8493 | 0.8653 | 0.0417 | 2.9999 |
| `region_adaptive_no_graph-seed87-a9f8bc67189e` | 87 | 2 | 0.0682 | 0.8811 | 0.8502 | 0.8644 | 0.0521 | 2.9999 |
| `sart_attention_pool-seed42-8a0d93da94fb` | 42 | 2 | 0.0682 | 0.8756 | 0.8500 | 0.8647 | 0.0521 | 3.0001 |
| `sart_coarse_ontology-seed42-18458baae240` | 42 | 2 | 0.0682 | 0.8767 | 0.8504 | 0.8681 | 0.0521 | 2.9999 |
| `sart_no_language_pair-seed42-be7b6b3251a3` | 42 | 2 | 0.0683 | 0.8776 | 0.8529 | 0.8683 | 0.0417 | 2.9997 |
| `sart_no_message-seed42-e1281e9d3b3a` | 42 | 2 | 0.0682 | 0.8742 | 0.8487 | 0.8634 | 0.0417 | 2.9998 |
| `sart_no_scalar-seed42-cd443a62a978` | 42 | 2 | 0.0682 | 0.8731 | 0.8481 | 0.8640 | 0.0417 | 3.0001 |
| `shuffled_structure-seed42-64b84413c2f7` | 42 | 2 | 0.0682 | 0.8742 | 0.8473 | 0.8639 | 0.0312 | 3.0000 |
| `structure_only-seed13-5f2f75e83b5e` | 13 | 2 | 0.0682 | 0.8544 | 0.8166 | 0.8289 | 0.0833 | 1.0000 |
| `structure_only-seed42-3eec9731aa46` | 42 | 2 | 0.0682 | 0.8771 | 0.8515 | 0.8670 | 0.0417 | 1.0000 |
| `structure_only-seed87-26cd2dfa499b` | 87 | 2 | 0.0682 | 0.8806 | 0.8517 | 0.8647 | 0.0521 | 1.0000 |
| `uniform_region_loop-seed13-b116c72786af` | 13 | 2 | 0.0682 | 0.8565 | 0.8180 | 0.8365 | 0.0729 | 6.0000 |
| `uniform_region_loop-seed42-28aeabfaf913` | 42 | 2 | 0.0682 | 0.8795 | 0.8513 | 0.8695 | 0.0417 | 6.0000 |
| `uniform_region_loop-seed87-c9a6950fd974` | 87 | 2 | 0.0682 | 0.8773 | 0.8488 | 0.8601 | 0.0521 | 6.0000 |
| `uniform_token_loop-seed13-0dd1d6c499a2` | 13 | 2 | 0.0682 | 0.8571 | 0.8196 | 0.8334 | 0.0625 | 6.0000 |
| `uniform_token_loop-seed42-80a4b3c4e0cc` | 42 | 2 | 0.0681 | 0.8767 | 0.8516 | 0.8677 | 0.0417 | 6.0000 |
| `uniform_token_loop-seed87-c2edb25d89d3` | 87 | 2 | 0.0682 | 0.8793 | 0.8486 | 0.8628 | 0.0521 | 6.0000 |

## Interpretation guardrails

- Validation lexical metrics are diagnostics; validation target loss alone selects checkpoints.
- External execution Pass@1, paired uncertainty estimates, and matched end-to-end compute costs are reported separately in the final research report.
- The report summarizes the 40 canonical Qwen2.5-Coder-7B runs. The authoritative registry and per-run histories remain the source data.
