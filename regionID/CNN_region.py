import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
import csv
import random
import numpy as np


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU

# Ensures that CUDA convolution operations are deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Dataset for Region ID ===
class RegionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['filename']
        region_id = int(row['Region_ID']) - 1  # Convert to 0-based index

        image_path = os.path.join(self.image_dir, file_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, region_id

# === Transforms ===
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),   # Random crop & resize
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = RegionDataset('labels_train.csv', 'images_train', transform_train)
val_dataset = RegionDataset('labels_val.csv', 'images_val', transform_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# === Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = efficientnet_b0(weights=None)  # Do not load torchvision weights
checkpoint = torch.load("efficientnet_b0_rwightman-3dd342df.pth", map_location=device)
model.load_state_dict(checkpoint)

model.classifier[1] = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.classifier[1].in_features, 15)
)

model = model.to(device)

# === Loss & Optimizer ===
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-4)
# === Learning Rate Scheduler ===
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.4, verbose=True)

best_val_accuracy = 0.0  # Track the best accuracy

for epoch in range(30):
    # === Training ===
    model.train()
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # === Validation ===
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            _, predicted = torch.max(preds, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Accuracy: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'efficientnet_region_classifier_best.pth')
        print(f"✅ Saved new best model with accuracy {val_accuracy:.2f}% at epoch {epoch}")

    # scheduler.step(val_accuracy)  # Uncomment if using scheduler

# === Evaluation ===
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        _, predicted = torch.max(preds, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

val_accuracy = 100 * correct / total
print(f"Validation Accuracy: {val_accuracy:.2f}%")


# === Save Predictions to CSV (369 val + 369 dummy) ===
val_predictions = []
idx_counter = 0

model.eval()
with torch.no_grad():
    for imgs, _ in val_loader:  # Labels not needed here
        imgs = imgs.to(device)
        preds = model(imgs)
        _, predicted = torch.max(preds, 1)

        # Add predicted Region_IDs (convert back to 1-based)
        for pred in predicted.cpu().numpy():
            val_predictions.append({
                'id': idx_counter,
                'Region_ID': int(pred) + 1
            })
            idx_counter += 1

# Append 369 dummy Region_ID=1 entries for the test set
for _ in range(369):
    val_predictions.append({
        'id': idx_counter,
        'Region_ID': 1
    })
    idx_counter += 1

# Save to CSV
with open('region_predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'Region_ID']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(val_predictions)

print(f"Region predictions saved to region_predictions.csv | Total rows: {len(val_predictions)}")

# Save the model's state dict
torch.save(model.state_dict(), 'efficientnet_region_classifier.pth')
print("Model saved to efficientnet_region_classifier.pth")
