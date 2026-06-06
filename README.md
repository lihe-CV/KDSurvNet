# KDSurvNet

**Imaging-Derived Deep Learning Inference of the Tumor Immune Microenvironment Predicts Survival and Immunotherapy Response in Gastric Cancer: A Multicenter Study**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

KDSurvNet is a biology-privileged deep learning framework that enables non-invasive inference of tumor immune microenvironment (TIME) for survival prediction and immunotherapy response assessment in gastric cancer. By transferring biological knowledge from invasive TIME biomarkers to routine CT imaging through cross-modal knowledge distillation, KDSurvNet provides a scalable and deployable solution for precision immuno-oncology.

**Key Innovation**: This work establishes a "virtual biopsy" strategy that leverages knowledge distillation to transfer tissue-derived immune information into radiological imaging, eliminating the need for invasive immunohistochemistry (IHC) at inference time.

---

## 🔬 Study Design

This multicenter observational study involved **5,538 patients** across **12 independent cohorts** from **5 cancer centers**:

- **Retrospective validation**: 4,180 gastric cancer patients
- **Prospective validation**: 179 gastric cancer patients  
- **Cross-cancer generalizability**: 766 colorectal cancer patients (436 colon, 330 rectal)
- **Immunotherapy response prediction**: 413 patients (352 + 61 in two independent cohorts)

---

## ✨ Key Features

- **Knowledge Distillation Framework**: Transfers biological information from Teacher network (with IHC markers) to Student network (CT + clinical data only)
- **Multi-modal Fusion**: Integrates imaging features (1000-dim), radiomics (584-dim), and clinical data (9-dim)
- **Robust Performance**: 
  - 5-year DFS: AUC 0.731 (95% CI: 0.709-0.754)
  - 5-year OS: AUC 0.747 (95% CI: 0.728-0.767)
  - Prospective validation DFS: AUC 0.739 (95% CI: 0.666-0.811)
- **Immunotherapy Response Prediction**: Significantly outperforms PD-L1 CPS (AUC: 0.756-0.761 vs. 0.615-0.632)
- **Cross-cancer Generalizability**: Validated in colorectal cancer cohorts

---

## 🏗️ Architecture

### Teacher Model: `ImageClinicalIHCBasedSurv`
- **Input**: CT images + Radiomics + Clinical features + **IHC markers (8-dim)**
- **Purpose**: Learns rich biological representations from invasive biomarkers

### Student Model: `ImageClinicalBasedSurv` (Deployable)
- **Input**: CT images + Radiomics + Clinical features (No IHC required)
- **Training**: Supervised by Teacher model through knowledge distillation
- **Deployment**: Inference using only non-invasive data

### Knowledge Distillation Strategy
- KL divergence loss on prediction logits (temperature T=10)
- Feature-level distillation (L2 loss)
- Progressive distillation factor: 0 → 1 (epoch 2000-4000)

*[Figure: Insert architecture diagram here]*

---

## 📊 Dataset Structure

```
dataset/
└── SMU/
    ├── data/
    │   ├── 1.h5          # Patient data in HDF5 format
    │   ├── 2.h5
    │   └── ...
    ├── two_cls_split.csv  # Train/val split for 2-class task
    └── four_cls_split.csv # Train/val split for 4-class task
```

**Data Format**: Each `.h5` file contains:
- CT image patches (224×224)
- Radiomics features (584-dim)
- Clinical features (9-dim)
- IHC markers (8-dim, for teacher training only)
- Survival labels (time, event)

---

## 🚀 Getting Started

### Requirements

```bash
Python >= 3.7
PyTorch >= 1.7.0
CUDA >= 10.2 (recommended)
```

**Dependencies**:
```bash
torch
torchvision
numpy
pandas
h5py
opencv-python
scikit-learn
scipy
lifelines
tensorboardX
tqdm
matplotlib
Pillow
```

### Installation

```bash
git clone https://github.com/AI4Onc/KDSurvNet.git
cd KDSurvNet
pip install -r requirements.txt  # Create this file based on dependencies above
```

---

## 💻 Usage

### 1. Train Teacher Model (with IHC)

```bash
cd code
python train_teacher.py \
    --root_path ../dataset/SMU/ \
    --exp ImageClinicalIHCBasedSurv_DFS \
    --model resnet18 \
    --num_classes two \
    --batch_size 48 \
    --max_iterations 6000 \
    --base_lr 0.0001
```

### 2. Train Student Model via Knowledge Distillation

```bash
python train_kdsurvnet.py \
    --root_path ../dataset/SMU/ \
    --exp ImageClinicalBasedSurv_KD_DFS \
    --model resnet18 \
    --num_classes two \
    --batch_size 24 \
    --max_iterations 6000 \
    --base_lr 0.0001
```

**Key Parameters**:
- `--root_path`: Path to dataset directory
- `--exp`: Experiment name for logging
- `--num_classes`: `"two"` (2-class) or `"four"` (4-class risk stratification)
- `--max_iterations`: Total training iterations (default: 6000)
- `--batch_size`: Batch size per GPU

### 3. Inference (Internal Dataset)

```bash
python inference.py \
    --data_path ../dataset/SMU/data \
    --checkpoint_path ../final_models/image+clinical/model.pth \
    --csv_file ../dataset/SMU/two_cls_split.csv \
    --split val \
    --save_path results_val.csv
```

### 4. Inference (External Dataset)

```bash
# Without IHC markers (deployable)
python inference_external.py \
    --data_path /path/to/external/data \
    --checkpoint_path ../final_models/image+clinical/model.pth

# With IHC markers (for comparison)
python inference_external_ihc.py \
    --data_path /path/to/external/data \
    --checkpoint_path ../final_models/image+clinical+ihc/model.pth
```

---

## 📁 Project Structure

```
KDSurvNet/
├── code/
│   ├── train_teacher.py              # Train teacher model with IHC
│   ├── train_kdsurvnet.py            # Train student model via KD
│   ├── train_baseline.py             # Train baseline models
│   ├── inference.py                  # Inference on internal dataset
│   ├── inference_external.py         # Inference on external dataset (no IHC)
│   ├── inference_external_ihc.py     # Inference on external dataset (with IHC)
│   ├── eval.py                       # Evaluation metrics
│   ├── eval_os.py                    # Overall survival evaluation
│   ├── losses.py                     # Custom loss functions
│   ├── nets/
│   │   ├── resnet.py                 # Model architectures
│   │   ├── vit.py                    # Vision Transformer (alternative)
│   │   └── configs.py                # Model configurations
│   └── dataloader/
│       ├── dataset.py                # Dataset and data augmentation
│       └── data_prepare.ipynb        # Data preprocessing notebook
├── dataset/
│   └── SMU/
│       ├── data/                     # Patient data (HDF5 format)
│       ├── two_cls_split.csv         # 2-class train/val split
│       └── four_cls_split.csv        # 4-class train/val split
├── manuscript/
│   └── KDSurvNet.pdf                 # Research manuscript (44 pages)
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 📈 Performance Summary

### Survival Prediction (Retrospective Multicenter Validation)

| Metric | AUC | 95% CI |
|--------|-----|--------|
| 5-year DFS | 0.731 | 0.709-0.754 |
| 5-year OS | 0.747 | 0.728-0.767 |

### Prospective Validation

| Metric | AUC | 95% CI |
|--------|-----|--------|
| 5-year DFS | 0.739 | 0.666-0.811 |
| 5-year OS | 0.788 | 0.720-0.856 |

### Immunotherapy Response Prediction

| Method | AUC Range |
|--------|-----------|
| KDSurvNet | 0.756-0.761 |
| PD-L1 CPS (standard) | 0.615-0.632 |

**Objective Response Rate**: 82.4% (KDSurvNet-high) vs. 14.5% (KDSurvNet-low)

*[Figure: Insert performance curves and risk stratification plots here]*

---

## 🔑 Key Components

### Survival Analysis
- **Time intervals**: [0, 8, 12, 19, 33, 115] months
- **Loss functions**: Negative log-likelihood (Cox proportional hazards)
- **Evaluation metric**: Concordance index (C-index)

### Knowledge Distillation
- **Temperature**: T = 10 for softmax distillation
- **Loss components**:
  - Cox NLL loss (survival prediction)
  - KL divergence loss (knowledge transfer)
  - L2 feature distillation loss
  - L2 regularization (α = 0.1)

### Training Strategy
- **Backbone**: ResNet-18 (pretrained on ImageNet)
- **Optimizer**: SGD (momentum=0.9, weight_decay=0.0001)
- **Learning rate**: 0.0001 with polynomial decay
- **Progressive KD**: Factor increases from 0 to 1 during epochs 2000-4000

---

## 📝 Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{li2026kdsurvnet,
  title={Imaging-Derived Deep Learning Inference of the Tumor Immune Microenvironment Predicts Survival and Immunotherapy Response in Gastric Cancer: A Multicenter Study},
  author={Li, He and Zhang, Taojun and Li, Junmeng and Wang, Wei and Huang, Weicai and Wang, Hongqiu and Li, Zihan and Yuan, Qingyu and Cheng, Chuanli and Xi, Sujuan and Han, Zhen and Wu, Lin and Feng, Wanying and Sun, Zepang and Yu, Jiang and Xiong, Wenjun and Chen, Tao and Li, Guoxin and Li, Zhenhui},
  journal={[Journal Name]},
  year={2026},
  note={Manuscript under review}
}
```

---

## 📧 Contact

**Correspondence**:
- Tao Chen: drchentao@163.com
- Guoxin Li: gzliguoxin@163.com  
- Zhenhui Li: lizhenhui@kmmu.edu.cn

**Institutions**:
- Department of General Surgery, Nanfang Hospital, Southern Medical University
- Department of Radiology, Yunnan Cancer Hospital, Kunming Medical University
- Multiple collaborating cancer centers (see manuscript for full list)

---

## 🙏 Acknowledgments

This work was supported by:
- Outstanding Youth Science Foundation of Yunnan Basic Research Project (202401AY070001-316)
- National Natural Science Foundation of China (82360345, 82001986)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 AI4Onc**

---

## ⚠️ Disclaimer

This software is provided for research purposes only. It has not been approved for clinical use. Future prospective interventional studies are warranted to confirm clinical utility before deployment in clinical practice.

---

## 🔄 Updates

- **2026-06**: Initial release with manuscript submission
- Code and example dataset published
- Pretrained models available upon request

---

*For detailed methodology, experimental results, and clinical implications, please refer to the [manuscript](manuscript/KDSurvNet.pdf).*
