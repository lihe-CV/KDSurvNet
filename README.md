# KDSurvNet

**Imaging-Derived Deep Learning Inference of the Tumor Immune Microenvironment Predicts Survival and Immunotherapy Response in Gastric Cancer: A Multicenter Study**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Overview

KDSurvNet is a biology-privileged deep learning framework that enables non-invasive inference of tumor immune microenvironment (TIME) for survival prediction and immunotherapy response assessment in gastric cancer. By transferring biological knowledge from invasive TIME biomarkers to routine CT imaging through cross-modal knowledge distillation, KDSurvNet provides a scalable and deployable solution for precision immuno-oncology.

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
- **Multi-modal Fusion**: Integrates imaging features, radiomics, and clinical data
- **Robust Performance**: 
  - 5-year DFS: AUC 0.731 (95% CI: 0.709-0.754)
  - 5-year OS: AUC 0.747 (95% CI: 0.728-0.767)
  - Prospective validation DFS: AUC 0.739 (95% CI: 0.666-0.811)
- **Immunotherapy Response Prediction**: Significantly outperforms PD-L1 CPS (AUC: 0.756-0.761 vs. 0.615-0.632)
- **Cross-cancer Generalizability**: Validated in colorectal cancer cohorts

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
