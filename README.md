# Improving protein and protein interactions using pseudo-dimers derived from monomeric proteins
This repository contains the official codebase for the paper:
**Improving protein and protein interactions using pseudo-dimers derived from monomeric proteins** (Nature Communications 2026)


---

## Introduction

SMP (Split-and-Merge Proxy) is a simple yet effective proxy strategy designed for pre-training protein interaction models. 

The overall framework consists of two stages: **Split** and **Merge**.

1. Split Stage: Each monomeric protein is partitioned to form a pseudo-dimer composed of two subsequences with corresponding structural information.

2. Merge Stage: Pairwise pseudo-dimers are embedded as the input for sequence-based or structure-based interaction models. The models are trained using labels derived from the split procedure, enabling task-specific pre-training for interaction prediction. 

Subsequently, the pre-trained models are fine-tuned on real protein dimer datasets without architectural modification, serving as strong initializations for downstream task.

---

## Downstream Task

### Protein-Protein Contact Map Prediction

Please refer to the `smp-contact` folder.

### Protein-Protein Docking

Please refer to the `smp-docking` folder.

### Protein-Protein Interaction

Please refer to the `smp-ppi` folder.

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{Du2026Improving,
  title={Improving protein and protein interactions using pseudo-dimers derived from monomeric proteins},
  author={Du, Hao and Zheng, Xinzhe and Ren, Yuchen and Huang, He and Gong, Xinqi and Ouyang, Wanli and Zhang, Yang and Lu, Yan},
  journal={Nature Communications},
  year={2026}
}
```