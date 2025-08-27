## Model Type and Training Strategy

**Model Used**: 

ConvNeXt-Tiny, a Convolutional Neural Network (CNN) architecture.

**Pre-trained**: 

Yes, the model is initialized with pre-trained ImageNet weights (ConvNeXt_Tiny_Weights.DEFAULT).

**Fine-Tuning**:

 The model’s final layer (model.classifier[2]) is replaced with a new nn.Linear layer to output two values: latitude and longitude. The entire model is fine-tuned on the location prediction task.

## Preprocessing Techniques

**Outlier Removal**:

Latitude and longitude values are filtered using the 1st and 99th quantiles to remove extreme geographic outliers.
(Implemented via the remove_outliers_quantile() function.)

**Data Normalization**:

Mean and standard deviation are computed from the dataset itself to normalize images using transforms.Normalize(mean, std).

**Image Augmentation (for Training)**:

#### Resize to 256×256

#### RandomResizedCrop (to 224×224)

#### Horizontal Flip

#### Random Rotation (±30°)

#### ColorJitter (Brightness, Contrast, Saturation, Hue)

#### Convert to Tensor and Normalize

**Validation Transformations**:

Only resizing, tensor conversion, and normalization are used (no augmentation), ensuring consistency during evaluation.

## Innovative / Noteworthy Ideas

**Location Regression from Images**:

The task is predicting geographic coordinates (latitude & longitude) from image content — a non-traditional regression task using CNNs.

**Custom Exclusion Handling**: 

Specific validation indices (e.g., [95, 145–161]) are excluded during validation and prediction by masking predictions to [0, 0] for those samples.

**Rounded Predictions for Output CSV**:

 Predicted latitude/longitude values are rounded to integers before saving, which may align with coarse geolocation needs.

**Robust Training Setup**:

Reproducibility ensured via consistent seeding across random, numpy, and torch.

Scheduler (ReduceLROnPlateau) adjusts learning rate based on validation loss plateau.

Model weights are saved only when validation loss improves.
