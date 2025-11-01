## 🧭 Overview

This module trains a **ConvNeXt-Tiny** model to predict **latitude** and **longitude** from campus images.  
Since the outputs are continuous values, the model performs **regression** using **Mean Squared Error (MSE)** loss.  

The workflow includes:
- Data preprocessing and augmentation  
- Model training and validation  
- Exclusion of specific problematic indices  
- Export of predictions in CSV format for evaluation/submission  

---

## ⚙️ Features

- **Quantile-based outlier removal** for target coordinates  
- **Automatic computation** of dataset mean and standard deviation for normalization  
- **Comprehensive data augmentation**: random crop, horizontal flip, rotation, and color jitter  
- **Transfer learning** using pretrained **ConvNeXt-Tiny (ImageNet weights)**  
- **Regression setup** with MSE loss, Adam optimizer, and ReduceLROnPlateau learning rate scheduler  

---

## 📁 Files

| File Name | Description |
|------------|-------------|
| `latlong_train.py` | Main training and prediction script |
| `latlong.pth` | Saved model weights (for testing or inference) |
| `Latlong_predictions.csv` | Generated CSV containing latitude–longitude predictions |

---

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install torch torchvision pandas numpy pillow
````

### 2. Run the Script

```bash
python latlong_train.py
```

---

## 🧩 Data

The input CSV must contain the following columns:

```
filename, latitude, longitude
```

The **`LocationDataset`** class:

* Loads images and returns `(image_tensor, tensor([lat, lon]))`
* Supports optional **quantile-based outlier filtering** via `filter_outliers=True`, which drops entries outside the 1st–99th percentile range

---

## 🧮 Data Transforms

### Training

* Resize
* RandomResizedCrop(224)
* RandomHorizontalFlip
* RandomRotation(±30°)
* ColorJitter
* ToTensor
* Normalize(mean, std)

### Validation

* Resize
* ToTensor
* Normalize(mean, std)

The mean and standard deviation are computed dynamically from the dataset using `compute_mean_std`.

---

## 🧠 Model

* **Base Model:** `convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)`
* **Modified Head:** Final classifier replaced with `nn.Linear(in_features, 2)` for predicting `(latitude, longitude)`

---

## 🔧 Training Setup

| Component         | Details                                     |
| ----------------- | ------------------------------------------- |
| **Loss Function** | `nn.MSELoss()`                              |
| **Optimizer**     | `Adam(lr=0.005, weight_decay=1e-4)`         |
| **Scheduler**     | `ReduceLROnPlateau(patience=3, factor=0.1)` |
| **Batch Size**    | 20                                          |

---

## 🧪 Validation Details

During validation and prediction, specific indices are **excluded** (set to zero values):

```
{95, 145, 146, 158, 159, 160, 161}
```

These entries are considered invalid or problematic.

---

## 📊 Output Format

### Generated File: `Latlong_predictions.csv`

| Column      | Description         |
| ----------- | ------------------- |
| `id`        | Image index         |
| `Latitude`  | Predicted latitude  |
| `Longitude` | Predicted longitude |

* Excluded indices are written as `0, 0`
* The script ensures the CSV file is **padded up to 738 rows** if necessary

Example:

```csv
id,Latitude,Longitude
0,219632, 143522
1,38336, 258
...
```

---

## 🏁 Outputs

After running the script:

* ✅ `latlong.pth` — best model weights
* ✅ `Latlong_predictions.csv` — final predictions for submission

---

## 🧾 Summary

This ConvNeXt-based latitude–longitude prediction pipeline combines **robust preprocessing**, **transfer learning**, and **careful validation handling** to deliver reliable geolocation estimates from images.
