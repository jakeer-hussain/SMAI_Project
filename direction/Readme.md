````markdown
# 🎯 Angle Estimation — ResNet18 with Angular Loss

**Module:** Orientation (Angle) Prediction  
**Project:** SMAI Project Phase

---

## 🧭 Overview

This module trains a **ResNet18** model to predict the **camera orientation angle (0°–360°)** of images.  
The goal is to infer the direction or facing angle of the camera based on the visual cues in the image.

Unlike standard regression, this model uses a **custom angular loss** designed to handle the **cyclic nature of angles**, ensuring that values like `1°` and `359°` are treated as close rather than distant.  
This approach leads to smoother optimization and more accurate orientation prediction.

---

## ⚙️ Features

- Custom **Angular Loss** function that respects 360° periodicity  
- Robust **data augmentation** with translation, rotation, and normalization  
- Transfer learning using **ResNet18 pretrained on ImageNet**  
- **Mean Angular Error (MAE)** as the main evaluation metric  
- Adaptive **learning rate scheduler** for smoother convergence  
- Automated CSV prediction export for validation and submission  

---

## 📁 Files

| File Name | Description |
|------------|-------------|
| `angle_train.py` | Main training, validation, and prediction script |
| `best_angle_classifier.pth` | Saved best-performing model weights |
| `angle_predictions.csv` | CSV file with predicted angles |

---

## 🧩 Dataset

Expected **CSV format** for training and validation:

| filename | angle |
|-----------|--------|
| img_001.jpg | 45 |
| img_002.jpg | 270 |
| ... | ... |

- **Images directories:**
  - `images_train/` → training data  
  - `images_val/` → validation data  
- Angles are measured in degrees **[0, 360)**

---

## 🧠 Model Architecture

- **Base model:** `ResNet18` (from `torchvision.models`)  
- **Weights:** Pretrained (`ResNet18_Weights.DEFAULT`)  
- **Head:** Single neuron linear layer (regression output)  

### Training Configuration

| Component | Setting |
|------------|----------|
| **Loss Function** | Custom `AngularLoss` |
| **Optimizer** | `Adam(lr=0.001)` |
| **Scheduler** | `ReduceLROnPlateau(patience=3, factor=0.5)` |
| **Batch Size** | 32 |
| **Epochs** | 30 |

---

## 🧮 Angular Loss Function

```python
class AngularLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.remainder(pred, 360)
        target = torch.remainder(target, 360)
        diff = torch.abs(pred - target)
        angular_diff = torch.minimum(diff, 360 - diff)
        return torch.mean(angular_diff)
````

This custom loss ensures that the model learns **the shortest circular distance** between predicted and true angles, avoiding large penalties across the 0°/360° boundary.

---

## 🧪 Training Pipeline

1. Load training and validation datasets from CSV
2. Apply resizing, normalization, and affine transformations
3. Train ResNet18 using **AngularLoss**
4. Compute **Mean Angular Error (MAE)** on the validation set
5. Save the model when the validation MAE improves

### Example Training Log

```
Epoch 10 | Train Loss: 12.8321 | Val MAE: 9.2543
✅ Best model saved with Val MAE: 9.2543
```

---

## 📊 Validation Results (Example)

| Metric                   | Value     |
| ------------------------ | --------- |
| Mean Angular Error (MAE) | **8.86°** |

*(Replace this with your actual results after training.)*

---

## 📄 Prediction Generation

After training, the script generates:

```
angle_predictions.csv
```

### Format

| id  | angle |
| --- | ----- |
| 0   | 180   |
| 1   | 45    |
| ... | ...   |

* Contains predictions for all validation images
* Includes **369 dummy rows (angle=0)** for test data
* Total of **738 rows** for submission compatibility

---

## 🚀 Usage

### 1️⃣ Install Dependencies

```bash
pip install torch torchvision pandas pillow numpy
```

### 2️⃣ Run Training

```bash
python angle_train.py
```

### 3️⃣ Outputs Generated

* ✅ `best_angle_classifier.pth` — best model checkpoint
* ✅ `angle_predictions.csv` — predictions for validation/test set

---

## 🧩 Notes

* Random seed fixed at **42** for reproducibility
* Deterministic cuDNN backend ensures consistent GPU runs
