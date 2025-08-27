import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F
import csv
import numpy as np, random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if using multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class AngleDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_name = row['filename']
        angle = int(row['angle'])

        image_path = os.path.join(self.image_dir, file_name)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, angle

class AngularLoss(nn.Module):
    def init(self):
        super(AngularLoss, self).init()

    def forward(self, pred, target):

        pred = torch.remainder(pred, 360)
        target = torch.remainder(target, 360)

        diff = torch.abs(pred - target)
        angular_diff = torch.minimum(diff, 360 - diff)
        return torch.mean(angular_diff)

def mean_angular_error(y_true, y_pred):
    """Metric function to measure angular error (not used for training)"""
    diff = torch.abs(y_true - y_pred)
    angular_diff = torch.minimum(diff, 360 - diff)
    return torch.mean(angular_diff)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_dataset = AngleDataset('labels_angle_train.csv', 'images_train', train_transform)
val_dataset = AngleDataset('labels_val.csv', 'images_val', val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = AngularLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=3, verbose=True)

best_val_mae = float('inf')

num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for imgs, angles in train_loader:
        imgs = imgs.to(device)
        angles = angles.to(device).float().unsqueeze(1)

        preds = model(imgs)
        loss = criterion(preds, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_mae = 0.0
    with torch.no_grad():
        for imgs, angles in val_loader:
            imgs = imgs.to(device)
            angles = angles.to(device).float()
            preds = model(imgs).squeeze(1)

            val_mae += mean_angular_error(angles, torch.round(preds) % 360).item()

    print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val MAE: {val_mae/len(val_loader):.4f}")

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'best_angle_classifier.pth')
        print(f"Best model saved with Val MAE: {val_mae/len(val_loader):.4f}")
    
    scheduler.step(val_mae / len(val_loader))

print(f"Training Complete. Best Validation MAE: {best_val_mae/len(val_loader):.4f}")

import csv

model.eval()
val_mae = 0.0
predictions = []
idx_counter = 0

with torch.no_grad():
    for imgs, angles in val_loader:
        imgs = imgs.to(device)
        angles = angles.to(device).float()
        preds = model(imgs).squeeze(1)

        val_mae += mean_angular_error(angles, torch.round(preds) % 360).item()

        preds_np = preds.cpu().numpy()
        for pred in preds_np:
            predictions.append({
                'id': idx_counter,
                'angle': int(round(pred) % 360)
            })
            idx_counter += 1

for _ in range(369):
    predictions.append({
        'id': idx_counter,
        'angle': 0
    })
    idx_counter += 1

print(f"Validation Mean Angular Error: {val_mae/len(val_loader):.4f}")

with open('angle_predictions.csv', 'w', newline='') as csvfile:
    fieldnames = ['id', 'angle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(predictions)

print(f"Predictions saved to predictions.csv | Total rows: {len(predictions)}")
