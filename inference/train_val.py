import sys
import pathlib

# Thêm thư mục gốc CoDeX-Net/ vào sys.path để import được model/
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
import time
import os
from tqdm import tqdm
import timm
from PIL import Image

from model.codex_net import CoDeX_Net

# Thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Đang sử dụng:", device)
print("Số GPU khả dụng:", torch.cuda.device_count())  # Kiểm tra số GPU

# Transform cho ResNet50
train_transform = transforms.Compose([
    #transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
val_transform = transforms.Compose([
    #transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Đường dẫn dữ liệu
train_dir = pathlib.Path('D:\LAB_AI\dataset_new\\racomData12class_chia\\train')
val_dir = pathlib.Path('D:\LAB_AI\dataset_new\\racomData12class_chia\\test')
# Load dataset từ hai thư mục riêng
train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

# DataLoader cho huấn luyện và test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)

# Lấy thông tin lớp
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Train samples: {len(train_dataset)}")       
print(f"Test samples: {len(val_dataset)}")
print(train_dataset.class_to_idx)

channels = (8, 8, 16, 16, 32, 32)
model = CoDeX_Net(channels_per_stage=channels, num_classes=12, drop_path_rate=0.1)
    
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")

# Hàm train và validate
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Train", colour="cyan", ncols=100) as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=running_loss/total, accuracy=f"{correct / total:.4f}")
            pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Test", colour="cyan", ncols=100) as pbar:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=running_loss / total, accuracy=f"{correct / total:.4f}")
                pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6)

start_epoch = 0
num_epochs = 60
best_acc = 0.0
best_epoch = start_epoch

resume_train = 'F:\THANHDAT\\final_model_epoch_53_acc_0.9020.pt'
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

if os.path.exists(resume_train):
    checkpoint = torch.load(resume_train, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint.get('val_acc', 0.0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    print(f"Resumed training from epoch {start_epoch} with val_acc: {best_acc:.4f}")
else:
    print(f"Checkpoint {resume_train} not found. Starting from scratch.")

# Train và validate
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    current_lr = optimizer.param_groups[0]['lr']

    start_time = time.time()
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    elapsed = time.time() - start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
    print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    print(f"time/epoch: {elapsed:.1f}s, current LR: {current_lr}")

    scheduler.step(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_path = f"best_model_epoch_{epoch+1}_acc_{best_acc:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1} with val_acc: {best_acc:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    final_model_path = f"final_model_epoch_{epoch+1}_acc_{val_acc:.4f}.pt"
    torch.save(checkpoint, final_model_path)
    print(f"💾 Saved final checkpoint at epoch {epoch+1} with val_acc: {val_acc:.4f}")
