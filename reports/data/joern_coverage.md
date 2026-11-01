# Joern structural extraction coverage

Counts are by unique source implementation (language + SHA256), not repeated translation directions.
Missing CFG/DFG are masked; no relations are imputed. CPG-node mapping coverage is pooled over nodes with offsets.

Unique source implementations: 28,447

| Language | Unique sources | Joern available | CFG available | DFG available | CPG node mapping | CFG region edges | DFG region edges |
|---|---:|---:|---:|---:|---:|---:|---:|
| Python | 9,073 | 9,073 | 9,073 | 9,073 | 81.07% (1,193,666/1,472,386) | 417,925 | 504,998 |
| Java | 9,660 | 9,654 | 9,654 | 9,654 | 100.00% (829,672/829,694) | 341,644 | 689,137 |
| C++ | 9,714 | 9,714 | 9,714 | 9,714 | 97.29% (1,092,850/1,123,260) | 460,429 | 879,016 |

## Directional row counts

- `test/C++-to-Java`: 794
- `test/C++-to-Python`: 756
- `test/Java-to-C++`: 794
- `test/Java-to-Python`: 759
- `test/Python-to-C++`: 756
- `test/Python-to-Java`: 759
- `train/C++-to-Java`: 8,224
- `train/C++-to-Python`: 7,696
- `train/Java-to-C++`: 8,224
- `train/Java-to-Python`: 7,638
- `train/Python-to-C++`: 7,696
- `train/Python-to-Java`: 7,638
- `validation/C++-to-Java`: 418
- `validation/C++-to-Python`: 389
- `validation/Java-to-C++`: 418
- `validation/Java-to-Python`: 392
- `validation/Python-to-C++`: 389
- `validation/Python-to-Java`: 392
