# External source-structure cache coverage

Generated from the checksum-verified task caches; no model scores are read.
Model-facing rows contain source text and allowed target interface only. Hidden tests and target implementations are absent.

| Benchmark | Set | Tasks | Problems | Directions | Source tokens | Mapped | Coverage | AST available | CFG available | DFG available |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| xcodeeval | common_core | 1992 | 332 | 6 | 825804 | 825778 | 99.9969% | 1922/1992 | 0/1992 | 0/1992 |

### xcodeeval: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 664 | 594 | 70 | 99.9941% (236546/236560) |
| Java | 664 | 664 | 0 | 99.9969% (391924/391936) |
| Python | 664 | 664 | 0 | 100.0000% (197308/197308) |

Cache config SHA256: `2940adbbfee154c659f33266703e7e032399c27ef540d09e8e7d6cff650d55a5`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| xcodeeval | full_directional | 3024 | 848 | 6 | 1215360 | 1215325 | 99.9971% | 2892/3024 | 0/3024 | 0/3024 |

### xcodeeval: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 1163 | 1031 | 132 | 99.9950% (458509/458532) |
| Java | 736 | 736 | 0 | 99.9975% (473453/473465) |
| Python | 1125 | 1125 | 0 | 100.0000% (283363/283363) |

Cache config SHA256: `10acfe10fad1bbc78536cc2bfeaaab6b8ecfa4c77ffa3c4a483df29b0cf0434b`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| humanevalx | common_core | 942 | 157 | 6 | 202206 | 202206 | 100.0000% | 942/942 | 0/942 | 0/942 |

### humanevalx: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 314 | 314 | 0 | 100.0000% (69204/69204) |
| Java | 314 | 314 | 0 | 100.0000% (74040/74040) |
| Python | 314 | 314 | 0 | 100.0000% (58962/58962) |

Cache config SHA256: `c151db655e8678f8f13a96ca02d393c93d2c4143b4b6ae53c399f242c908d49b`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| humanevalx | full_directional | 956 | 164 | 6 | 205422 | 205422 | 100.0000% | 956/956 | 0/956 | 0/956 |

### humanevalx: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 318 | 318 | 0 | 100.0000% (70265/70265) |
| Java | 321 | 321 | 0 | 100.0000% (75784/75784) |
| Python | 317 | 317 | 0 | 100.0000% (59373/59373) |

Cache config SHA256: `4606c0af272c1793d5f0aace64236710998073868ecedd8c1c660f7c926d85e5`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| transcoder_test | common_core | 1926 | 321 | 6 | 208312 | 208312 | 100.0000% | 1926/1926 | 0/1926 | 0/1926 |

### transcoder_test: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 642 | 642 | 0 | 100.0000% (69524/69524) |
| Java | 642 | 642 | 0 | 100.0000% (75144/75144) |
| Python | 642 | 642 | 0 | 100.0000% (63644/63644) |

Cache config SHA256: `83601cf016f6bb22551223a66232cd7e46a14fd0289d8657eeaf805276bcd346`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| transcoder_test | full_directional | 1936 | 326 | 6 | 210052 | 210052 | 100.0000% | 1936/1936 | 0/1936 | 0/1936 |

### transcoder_test: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| C++ | 646 | 646 | 0 | 100.0000% (70100/70100) |
| Java | 645 | 645 | 0 | 100.0000% (75946/75946) |
| Python | 645 | 645 | 0 | 100.0000% (64006/64006) |

Cache config SHA256: `3224f7e72260dc5abfaa81bd458e9a2a32e65e78ee571877e45cea44ef0439f0`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| avatar_tc | common_core | 3298 | 1649 | 2 | 692952 | 692952 | 100.0000% | 3298/3298 | 0/3298 | 0/3298 |

### avatar_tc: common_core

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| Java | 1649 | 1649 | 0 | 100.0000% (406998/406998) |
| Python | 1649 | 1649 | 0 | 100.0000% (285954/285954) |

Cache config SHA256: `88fd809e6ea9d652b267ee23ad6898a0ed42db77bf7cf650a0257d729f68df1d`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.

| avatar_tc | full_directional | 3298 | 1649 | 2 | 692952 | 692952 | 100.0000% | 3298/3298 | 0/3298 | 0/3298 |

### avatar_tc: full_directional

| Source language | Tasks | AST parsed | Fallbacks | Token mapping |
|---|---:|---:|---:|---:|
| Java | 1649 | 1649 | 0 | 100.0000% (406998/406998) |
| Python | 1649 | 1649 | 0 | 100.0000% (285954/285954) |

Cache config SHA256: `23843983ec69feebb973bfa83a0d0cf96336acd18fdd84222629c4b0ed508452`; tokenizer SHA256: `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
