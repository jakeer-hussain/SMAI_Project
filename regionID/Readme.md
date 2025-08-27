## Model Description (Type, Pretrained, Training Strategy, Innovations)

## Model Type and Training Strategy

### Architecture Used: 

EfficientNet-B0 — a Convolutional Neural Network (CNN) designed for image classification, known for its efficiency and accuracy trade-off.

### Pretrained Weights:

Not loaded from torchvision (weights=None).

Instead, custom pretrained weights from a .pth file (efficientnet_b0_rwightman-3dd342df.pth) are loaded using torch.load() — this likely comes from the RWightman repo.

### Fine-tuned:

The classifier's last layer is replaced (nn.Linear(..., 15)) to classify 15 distinct Region_IDs (converted to 0-based index).

The entire model is fine-tuned end-to-end on the new dataset using cross-entropy loss.

## Preprocessing Techniques

### Image Augmentation (Training Set)

Used to increase dataset diversity and prevent overfitting:

RandomResizedCrop(224, scale=(0.8, 1.0))

RandomHorizontalFlip()

RandomVerticalFlip(p=0.2)

RandomRotation(15)

ColorJitter (brightness, contrast, saturation, hue)

RandomAffine (rotation, translation, scaling)

ToTensor() and Normalize with ImageNet stats

## Validation Set Preprocessing

Minimal transformations to ensure evaluation consistency:

Resize((256, 256))

ToTensor() and Normalize

# Noteworthy Design Choices / Innovations

## Custom Weight Loading:

EfficientNet weights are loaded from a custom .pth file, providing more flexibility than torchvision’s default. This allows potentially better pretrained performance or compatibility with other frameworks.

## Label Smoothing:

nn.CrossEntropyLoss(label_smoothing=0.1) helps improve generalization and handle noisy labels.

## Seeding and Determinism:

Full seed setup for reproducibility across random, numpy, torch, and torch.cuda.

Ensures deterministic behavior with torch.backends.cudnn.deterministic = True.

## Learning Rate Scheduler (optional):

Included ReduceLROnPlateau scheduler (commented out) that can adaptively reduce LR on validation plateau — good for long training.

## Validation Strategy:

Tracks and saves only the best-performing model based on validation accuracy, ensuring optimal generalization.

## Prediction Export with test predictions padding:

It tests on test data after thoroughly improving the model using train and validation data to check the robustness and generalization of the model
