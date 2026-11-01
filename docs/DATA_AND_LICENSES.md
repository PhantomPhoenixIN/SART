# Data and license notices

This repository contains task-selection manifests derived from public benchmarks. Three compressed manifests include benchmark source programs and remain subject to their upstream terms. Consult the pinned upstream snapshot and its license before redistribution or reuse.

| Resource | Pinned source | Upstream terms noted for this artifact |
| --- | --- | --- |
| XLCoST | [XLCoST](https://github.com/reddy-lab-code-research/XLCoST), revision in `configs/data/artifacts.yaml` | Dataset terms must be checked against the downloaded archive; the archive is not bundled. |
| xCodeEval | [xCodeEval](https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval), pinned revision `aa09386604075a3e80bcdb315dd68bcc117dedc0` | Check the pinned dataset card and repository license. Credit the xCodeEval authors when using the derived manifests. |
| TransCoder-Test | [TransCoder](https://github.com/facebookresearch/TransCoder), revision `1c21887dc55a88ea6535e669952d419256fbc24a` | Creative Commons Attribution-NonCommercial 4.0; the source license is reproduced in `third_party_licenses/TransCoder-LICENSE`. |
| HumanEval-X | [HumanEval-X](https://huggingface.co/datasets/zai-org/humaneval-x), pinned revision in `configs/data/artifacts.yaml` | Apache-2.0 is recorded for the pinned source; verify the relevant dataset and evaluation-code terms upstream. |
| AVATAR / AVATAR-TC | [AVATAR](https://github.com/wasiahmad/AVATAR), revision `6ae262ec44178087974e8d57cb7c034a164547d6`; [CoTran](https://github.com/prithwishjana/cotran) for the TC adaptation | The AVATAR dataset is CC BY-SA 4.0. Check CoTran artifact terms separately. |
| Qwen2.5-Coder / StarCoder2 | Revisions in `configs/data/artifacts.yaml` | Model weights are not redistributed here. Follow the model licenses at their pinned repositories. |

The local execution outcomes are generated measurements of these benchmarks. Their interpretation depends on the pinned task manifests and the local compatibility harness. The repository's package metadata states that component dataset/model licenses apply; it does not grant additional rights over upstream material.
