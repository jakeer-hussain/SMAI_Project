# 🛰️ SMAI Project Phase-2 – Image-Based Geo-Localization

**Course:** Statistical Methods in Artificial Intelligence (SMAI)   

---

## 🌍 Overview

This project focuses on **predicting geographic and orientation attributes** from campus images.  
Each input image is used to predict four key values:

- **Latitude (scaled)**
- **Longitude (scaled)**
- **Angle** (camera orientation in degrees)
- **Region ID** (integer 1–15)

The models are trained and validated on labeled subsets, and evaluated using:
- **MSE** → Latitude, Longitude, and Angle  
- **Accuracy** → Region ID  

---

## 🧠 Methodology

Separate deep-learning models were designed for each prediction task using **PyTorch**.  
A range of state-of-the-art CNN and transformer architectures were explored and fine-tuned, including:

- **ConvNeXt-Tiny / Base**
- **Vision Transformer (ViT-B/16)**
- **EfficientNet-B0**
- **ResNet-18 / ResNet-50**

All experiments were conducted with reproducible seeds, standardized augmentations, and adaptive learning-rate schedulers.

---

### 1️⃣ Latitude & Longitude Prediction
- **Type:** Regression  
- **Architecture:** ViT-B/16 & ConvNeXt  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam (`lr = 0.001`)  
- **Scaling:** Latitude & longitude were normalized for stable training  
- **Validation MSE:** ≈ **1.16 × 10⁶ (averaged)**  

---

### 2️⃣ Angle Prediction
- **Type:** Circular Regression  
- **Architecture:** ResNet-18 / ConvNeXt-Tiny  
- **Loss Function:** Custom `AngularLoss`  
  (wraps angular values within 0–360° and minimizes circular distance)  
- **Metric:** Mean Angular Error (MAE)  
- **Validation MAE:** ≈ **0.0246** (≈ **8.9°**)  

---

### 3️⃣ Region ID Classification
- **Type:** Multi-class Classification (15 classes)  
- **Architecture:** EfficientNet-B0 (pretrained on ImageNet)  
- **Loss Function:** Cross-Entropy with Label Smoothing (0.1)  
- **Metric:** Accuracy  
- **Validation Accuracy:** ≈ **94.85%**

---

## ⚙️ Training Setup

- **Framework:** PyTorch  
- **Batch Size:** 32  
- **Epochs:** 50
- **Optimizer:** Adam / AdamW  
- **Scheduler:** ReduceLROnPlateau  
- **Hardware:** NVIDIA GPU (CUDA)  
- **Seed:** 42 (for full reproducibility)  

---

## 📁 Repository Structure

```

SMAI_Project/
├── latitude_longitude/
│   ├── train_latlon_vit_convnext.py
│   └── README.md
├── angle/
│   ├── angle_prediction_resnet.py
│   └── README.md
├── region/
│   ├── region_classifier_efficientnet.py
│   └── README.md
├── data/
│   ├── images_train/
│   ├── images_val/
│   ├── labels_train.csv
│   ├── labels_val.csv
└── README.md

```

---

## 🧩 Results Summary

| Task | Model | Metric | Validation Score |
|------|--------|---------|------------------|
| Latitude | ConvNeXt | MSE | ~1.16×10⁶ |
| Longitude | ViT-B/16 | MSE | ~1.16×10⁶ |
| Angle | ResNet-18 | MAE | 0.0246 (~8.9°) |
| Region ID | EfficientNet-B0 | Accuracy | 94.85% |

---

## 🚀 Future Work

- Multi-task unified model for joint learning of all four outputs  
- Use **Swin Transformer** or **ConvNeXt-V2** for improved spatial sensitivity  
- Incorporate attention-based regularization  

---
