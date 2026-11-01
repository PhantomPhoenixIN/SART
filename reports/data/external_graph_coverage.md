# External source-structure cache coverage

Generated from the checksum-verified task caches; no model scores are read.
Model-facing rows contain source text and allowed target interface only. Hidden tests and target implementations are absent.

| Benchmark | Set | Tasks | Problems | Directions | Source tokens | Mapped | Coverage | AST available | CFG available | DFG available |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xcodeeval | common_core | 1992 | 332 | 6 | 825804 | 825778 | 99.9969% | 1922/1992 | 1992/1992 | 1992/1992 |

### xcodeeval: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 664 | 594 | 70 | 99.9941% (236546/236560) |
| Java | 664 | 664 | 0 | 99.9969% (391924/391936) |
| Python | 664 | 664 | 0 | 100.0000% (197308/197308) |

AST manifest SHA256: `3623a49653d03f780fe0252d2fc231648c0fb7d7045a2a5713872c1a598f64cf`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| xcodeeval | full_directional | 3024 | 848 | 6 | 1215360 | 1215325 | 99.9971% | 2892/3024 | 3024/3024 | 3024/3024 |

### xcodeeval: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 1163 | 1031 | 132 | 99.9950% (458509/458532) |
| Java | 736 | 736 | 0 | 99.9975% (473453/473465) |
| Python | 1125 | 1125 | 0 | 100.0000% (283363/283363) |

AST manifest SHA256: `16917288fbc89cf932e1c7446cd7bc3388b7b177e2edb8735db59d8f59a8ccbb`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| humanevalx | common_core | 942 | 157 | 6 | 202206 | 202206 | 100.0000% | 942/942 | 942/942 | 942/942 |

### humanevalx: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 314 | 314 | 0 | 100.0000% (69204/69204) |
| Java | 314 | 314 | 0 | 100.0000% (74040/74040) |
| Python | 314 | 314 | 0 | 100.0000% (58962/58962) |

AST manifest SHA256: `bd6b65e772ac5ad13bf85324d7f96c9d213314425abe0ace6b2a98793fd2802e`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| humanevalx | full_directional | 956 | 164 | 6 | 205422 | 205422 | 100.0000% | 956/956 | 956/956 | 956/956 |

### humanevalx: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 318 | 318 | 0 | 100.0000% (70265/70265) |
| Java | 321 | 321 | 0 | 100.0000% (75784/75784) |
| Python | 317 | 317 | 0 | 100.0000% (59373/59373) |

AST manifest SHA256: `a2fd96c8c92b1756a0af9d2623ac4570eac4d613218b5f321bfa884a2e19c8b1`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| transcoder_test | common_core | 1926 | 321 | 6 | 208312 | 208312 | 100.0000% | 1926/1926 | 1926/1926 | 1926/1926 |

### transcoder_test: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 642 | 642 | 0 | 100.0000% (69524/69524) |
| Java | 642 | 642 | 0 | 100.0000% (75144/75144) |
| Python | 642 | 642 | 0 | 100.0000% (63644/63644) |

AST manifest SHA256: `ba90e764604a5d84cfccb6bb29ef8cb85f2958c226f1ec6a3e8f4c991c76506f`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| transcoder_test | full_directional | 1936 | 326 | 6 | 210052 | 210052 | 100.0000% | 1936/1936 | 1936/1936 | 1936/1936 |

### transcoder_test: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 646 | 646 | 0 | 100.0000% (70100/70100) |
| Java | 645 | 645 | 0 | 100.0000% (75946/75946) |
| Python | 645 | 645 | 0 | 100.0000% (64006/64006) |

AST manifest SHA256: `86e8db78f25e911cadf9750ddc027000b2939a6c9ec485615f0ccab19f53320a`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| avatar_tc | common_core | 3298 | 1649 | 2 | 692952 | 692952 | 100.0000% | 3298/3298 | 3298/3298 | 3298/3298 |

### avatar_tc: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| Java | 1649 | 1649 | 0 | 100.0000% (406998/406998) |
| Python | 1649 | 1649 | 0 | 100.0000% (285954/285954) |

AST manifest SHA256: `db7235ea827084a59ce0e26001826814f19637bab33a813d248d10755d5fa4a3`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.

| avatar_tc | full_directional | 3298 | 1649 | 2 | 692952 | 692952 | 100.0000% | 3298/3298 | 3298/3298 | 3298/3298 |

### avatar_tc: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| Java | 1649 | 1649 | 0 | 100.0000% (406998/406998) |
| Python | 1649 | 1649 | 0 | 100.0000% (285954/285954) |

AST manifest SHA256: `ae11c4a79e25679c5260da0f8c626ba0006510b17aea3da3d4d856cb8fd69995`; Joern manifest SHA256: `8b62803cf64c904eccff7faa580efd97f046ddc716504507e603f3af0490694c`.
