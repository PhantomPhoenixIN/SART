# Completed Confirmatory 7B In-Domain Validation Summary

Runs summarized: **17**

Descriptive validation metrics only. Not external executable evidence, not a significance analysis, and not a basis for model selection.

Latest recorded validation row is used for each completed run. The frozen two-epoch horizon is common across confirmatory 7B runs.

| Architecture | n | Seeds | Target loss | BLEU-4 | CodeBLEU | Exact match | CrystalBLEU |
|---|---:|---|---:|---:|---:|---:|---:|
| dense | 3 | 13, 42, 87 | 0.06837 | 0.87034 | 0.83775 | 0.06250 | 0.85357 |
| dense_param_match | 3 | 13, 42, 87 | 0.06811 | 0.87069 | 0.83754 | 0.05903 | 0.85501 |
| mor_style | 3 | 13, 42, 87 | 0.06817 | 0.86954 | 0.83890 | 0.05556 | 0.85190 |
| random_region | 3 | 13, 42, 87 | 0.06821 | 0.87008 | 0.83953 | 0.05556 | 0.85329 |
| shuffled_structure | 1 | 42 | 0.06825 | 0.87417 | 0.84734 | 0.03125 | 0.86392 |
| uniform_region_loop | 3 | 13, 42, 87 | 0.06818 | 0.87109 | 0.83933 | 0.05556 | 0.85536 |
| uniform_token_loop | 1 | 13 | 0.06820 | 0.85709 | 0.81959 | 0.06250 | 0.83341 |

Per-run details are available in the adjacent JSON artifact.
