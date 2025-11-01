# 🗺️ Region ID Classification — EfficientNet-B0

**Module:** Region ID Prediction  
**Project:** SMAI Project

---

## 🌐 Overview

This module trains an **EfficientNet-B0** model to classify campus images into **15 distinct regions (Region_ID 1–15)**.  
The model predicts which region of the IIIT Hyderabad campus an image belongs to, based purely on its visual features — effectively localizing photos within the campus environment.

---

## ⚙️ Features

- Custom **`RegionDataset`** class for image–region mapping.  
- Strong **data augmentation** using random rotation, flip, affine transform, and color jitter.  
- Transfer learning from pretrained **EfficientNet-B0** (timm weights).  
- **Label smoothing** for improved generalization and stability.  
- Optional **learning rate scheduling** for dynamic optimization.  
- **Automatic checkpointing** of the best-performing model.

---

## 📁 Files

| File Name | Description |
|------------|-------------|
| `region_train.py` | Main training, validation, and prediction script |
| `efficientnet_b0_rwightman-3dd342df.pth` | Pretrained EfficientNet-B0 weights |
| `efficientnet_region_classifier_best.pth` | Saved weights of best-performing model |
| `region_predictions.csv` | Generated predictions for validation/test data |

---

## 🧩 Dataset

The dataset should be provided in **CSV format** for both training and validation:

| filename    | Region_ID |
|-------------|-----------|
| img_001.jpg | 3         |
| img_002.jpg | 12        |
| ...         | ...       |

- `Region_ID` values range from **1–15** (internally converted to **0–14**).  
- Image folders:
  - `images_train/` → for training data  
  - `images_val/` → for validation data  

---

## 🧠 Model Architecture

- **Base model:** `EfficientNet-B0` (from `torchvision`)  
- **Weights:** Pretrained from timm (`efficientnet_b0_rwightman-3dd342df.pth`)  
- **Modified classifier head:**

```python
nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 15)
)
````

### Training Configuration

| Component      | Setting                                     |
| -------------- | ------------------------------------------- |
| **Loss**       | `CrossEntropyLoss(label_smoothing=0.1)`     |
| **Optimizer**  | `Adam(lr=0.0002, weight_decay=1e-4)`        |
| **Scheduler**  | `ReduceLROnPlateau` *(optional)* |
| **Batch Size** | 32                                          |
| **Epochs**     | 50                                          |

---

## 🧪 Training Pipeline

1. Load dataset using **`RegionDataset`**
2. Apply augmentations:

   * Random crop
   * Random horizontal flip
   * Rotation
   * Color jitter
   * Affine transform
3. Train EfficientNet-B0 with **CrossEntropy loss + label smoothing**
4. Evaluate validation accuracy after each epoch
5. Save model when validation accuracy improves

### Example Training Output

```
Epoch 12 | Train Loss: 0.9345 | Val Accuracy: 84.56%
✅ Saved new best model with accuracy 84.56% at epoch 12
```

---

## 📊 Validation Results

| Metric              | Value      |
| ------------------- | ---------- |
| Validation Accuracy | **94.85%** |

---

## 🧾 Prediction Generation

After training, predictions for the validation set are stored in:

```
region_predictions.csv
```

### Format

| id  | Region_ID |
| --- | --------- |
| 0   | 5         |
| 1   | 9         |
| ... | ...       |

* An additional **369 dummy entries (Region_ID=1)** are appended for the test set.
* Final CSV file contains **738 total rows**, maintaining submission compatibility.

---

## 🚀 Usage

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision pandas numpy pillow
```

### 2️⃣ Run Training

```bash
python region_train.py
```

### 3️⃣ Outputs Generated

* ✅ `efficientnet_region_classifier_best.pth`
* ✅ `efficientnet_region_classifier.pth`
* ✅ `region_predictions.csv`

---

## 🧩 Notes

* Training uses **deterministic seeding** (`seed=42`) for reproducibility.
* **CUDA deterministic flags** ensure stable behavior across multiple runs.

```python
scheduler.step(val_accuracy)
```

---

## 🏁 Summary

This module leverages **EfficientNet-B0** with transfer learning and advanced augmentations to accurately predict campus regions from images.
The approach ensures robustness, reproducibility, and seamless integration with evaluation platforms.
