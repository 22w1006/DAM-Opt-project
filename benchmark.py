import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import random

from Adan.adan import Adan

# 固定random seed用于复现
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 超参数接口
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ResNet-50 on CIFAR-10')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')  # Adam/Adan建议较小lr
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers')
    return parser.parse_args([])  # [] for Jupyter, remove for script

args = get_args()

# 数据预处理
print("==> Preparing data...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#trainset = torch.utils.data.Subset(trainset, np.random.choice(len(trainset), 10000, replace=False))  # 仅使用10000个样本
print(f"Trainset size: {len(trainset)}")
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#valset = torch.utils.data.Subset(valset, np.random.choice(len(valset), 2000, replace=False))  # 仅使用2000个样本
print(f"Valset size: {len(valset)}")
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

# 简单ResNet-50
from torchvision.models import resnet50

class ResNet50CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet50(num_classes=num_classes)
        # 修改第一层适应CIFAR-10
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
    def forward(self, x):
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Use device:", device)

# 创建模型副本
print("==> Building models...")
net_adam = ResNet50CIFAR10().to(device)
net_adan = ResNet50CIFAR10().to(device)
net_cam = ResNet50CIFAR10().to(device)

criterion = nn.CrossEntropyLoss()
optimizer_adam = optim.Adam(net_adam.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_adan = Adan(net_adan.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_dam = DAM(net_cam.parameters(), lr=args.lr)

print("==> Start training...")
train_loss_list_adam, val_loss_list_adam = [], []
train_acc_list_adam, val_acc_list_adam = [], []

train_loss_list_adan, val_loss_list_adan = [], []
train_acc_list_adan, val_acc_list_adan = [], []

train_loss_list_dam, val_loss_list_dam = [], []
train_acc_list_dam, val_acc_list_dam = [], []

# 新增：累计训练时间记录
cumulative_time_adam = []
cumulative_time_adan = []
cumulative_time_dam = []
total_time_adam = 0.0
total_time_adan = 0.0
total_time_dam = 0.0

# 记录达到各acc阈值的时间
acc_thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
def get_acc_time_dict():
    return {th: None for th in acc_thresholds}

adam_train_acc_time = get_acc_time_dict()
adam_val_acc_time = get_acc_time_dict()
adan_train_acc_time = get_acc_time_dict()
adan_val_acc_time = get_acc_time_dict()
dam_train_acc_time = get_acc_time_dict()
dam_val_acc_time = get_acc_time_dict()

# 新增：每10个epoch保存一次参数
save_dir = './saved_models'
os.makedirs(save_dir, exist_ok=True)

for epoch in range(args.epochs):
    print(f"\nEpoch [{epoch+1}/{args.epochs}]")

    # CAM
    start_time = time.time()
    net_cam.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)   
        optimizer_dam.zero_grad()
        outputs = net_cam(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_dam.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_list_dam.append(train_loss)
    train_acc_list_dam.append(train_acc)
    end_time = time.time()
    total_time_dam += end_time - start_time
    cumulative_time_dam.append(total_time_dam)
    print(f"  [DAM][Train] Epoch Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    # 记录train acc阈值时间
    for th in acc_thresholds:
        if dam_train_acc_time[th] is None and train_acc >= th:
            dam_train_acc_time[th] = total_time_dam

    net_cam.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_cam(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_loss_list_dam.append(val_loss)
    val_acc_list_dam.append(val_acc)
    print(f"  [DAM][Val] Epoch Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    # 记录val acc阈值时间
    for th in acc_thresholds:
        if dam_val_acc_time[th] is None and val_acc >= th:
            dam_val_acc_time[th] = total_time_dam

    # Adam
    start_time = time.time()
    net_adam.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_adam.zero_grad()
        outputs = net_adam(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_adam.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_list_adam.append(train_loss)
    train_acc_list_adam.append(train_acc)
    end_time = time.time()
    total_time_adam += end_time - start_time
    cumulative_time_adam.append(total_time_adam)
    print(f"  [Adam][Train] Epoch Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    # 记录train acc阈值时间
    for th in acc_thresholds:
        if adam_train_acc_time[th] is None and train_acc >= th:
            adam_train_acc_time[th] = total_time_adam

    net_adam.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_adam(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_loss_list_adam.append(val_loss)
    val_acc_list_adam.append(val_acc)
    print(f"  [Adam][Val] Epoch Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    # 记录val acc阈值时间
    for th in acc_thresholds:
        if adam_val_acc_time[th] is None and val_acc >= th:
            adam_val_acc_time[th] = total_time_adam

    # Adan
    start_time = time.time()
    net_adan.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer_adan.zero_grad()
        outputs = net_adan(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_adan.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    train_loss = running_loss / total
    train_acc = correct / total
    train_loss_list_adan.append(train_loss)
    train_acc_list_adan.append(train_acc)
    end_time = time.time()
    total_time_adan += end_time - start_time
    cumulative_time_adan.append(total_time_adan)
    print(f"  [Adan][Train] Epoch Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    # 记录train acc阈值时间
    for th in acc_thresholds:
        if adan_train_acc_time[th] is None and train_acc >= th:
            adan_train_acc_time[th] = total_time_adan

    net_adan.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(valloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net_adan(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    val_loss = val_loss / val_total
    val_acc = val_correct / val_total
    val_loss_list_adan.append(val_loss)
    val_acc_list_adan.append(val_acc)
    print(f"  [Adan][Val] Epoch Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
    # 记录val acc阈值时间
    for th in acc_thresholds:
        if adan_val_acc_time[th] is None and val_acc >= th:
            adan_val_acc_time[th] = total_time_adan

    # 每10个epoch保存一次参数
    if (epoch + 1) % 10 == 0:
        torch.save(net_adam.state_dict(), os.path.join(save_dir, f'net_adam_epoch{epoch+1}.pth'))
        torch.save(net_adan.state_dict(), os.path.join(save_dir, f'net_adan_epoch{epoch+1}.pth'))
        torch.save(net_cam.state_dict(), os.path.join(save_dir, f'net_dam_epoch{epoch+1}.pth'))
        print(f"Saved model parameters at epoch {epoch+1}")

    # 打印每个epoch的总结
    print(f"Summary Epoch {epoch+1}/{args.epochs} | Adam: Train Loss: {train_loss_list_adam[-1]:.4f} Acc: {train_acc_list_adam[-1]:.4f} | Val Loss: {val_loss_list_adam[-1]:.4f} Acc: {val_acc_list_adam[-1]:.4f}")
    print(f"                 | Adan: Train Loss: {train_loss_list_adan[-1]:.4f} Acc: {train_acc_list_adan[-1]:.4f} | Val Loss: {val_loss_list_adan[-1]:.4f} Acc: {val_acc_list_adan[-1]:.4f}")
    print(f"                 | DAM: Train Loss: {train_loss_list_dam[-1]:.4f} Acc: {train_acc_list_dam[-1]:.4f} | Val Loss: {val_loss_list_dam[-1]:.4f} Acc: {val_acc_list_dam[-1]:.4f}")


# 打印表格
import pandas as pd

def format_time(t):
    if t is None:
        return '/'
    else:
        return f"{t:.1f}s"

def build_acc_time_row(train_dict, val_dict):
    return [format_time(train_dict[th]) for th in acc_thresholds] + [format_time(val_dict[th]) for th in acc_thresholds]

columns = [f"Train@{int(th*100)}%" for th in acc_thresholds] + [f"Val@{int(th*100)}%" for th in acc_thresholds]
data = [
    build_acc_time_row(adam_train_acc_time, adam_val_acc_time),
    build_acc_time_row(adan_train_acc_time, adan_val_acc_time),
    build_acc_time_row(dam_train_acc_time, dam_val_acc_time),
]
index = ['Adam', 'Adan', 'DAM']
df = pd.DataFrame(data, columns=columns, index=index)
print("\nCumulative training time required to reach each accuracy threshold (unit: seconds, '/' means not reached):")
print(df)

# 画图并保存图片
import os

fig = plt.figure(figsize=(12,5))

plt.subplot(1,3,1)
plt.plot(train_loss_list_adam, label='Train Loss (Adam)')
plt.plot(val_loss_list_adam, label='Val Loss (Adam)')
plt.plot(train_loss_list_adan, label='Train Loss (Adan)')
plt.plot(val_loss_list_adan, label='Val Loss (Adan)')
plt.plot(train_loss_list_dam, label='Train Loss (DAM)')
plt.plot(val_loss_list_dam, label='Val Loss (DAM)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curve')

plt.subplot(1,3,2)
plt.plot(train_acc_list_adam, label='Train Acc (Adam)')
plt.plot(val_acc_list_adam, label='Val Acc (Adam)')
plt.plot(train_acc_list_adan, label='Train Acc (Adan)')
plt.plot(val_acc_list_adan, label='Val Acc (Adan)')
plt.plot(train_acc_list_dam, label='Train Acc (DAM)')
plt.plot(val_acc_list_dam, label='Val Acc (DAM)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')

# 新增：累计训练时间-准确率曲线（纵轴为acc，横轴为时间）
plt.subplot(1,3,3)
plt.plot(cumulative_time_adam, val_acc_list_adam, label='Adam')
plt.plot(cumulative_time_adan, val_acc_list_adan, label='Adan')
plt.plot(cumulative_time_dam, val_acc_list_dam, label='DAM')
plt.xlabel('Cumulative Training Time (s)')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.title('Validation Accuracy vs. Training Time')

plt.tight_layout()

# 保存图片
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
fig_path = os.path.join(save_dir, 'training_results.png')
plt.savefig(fig_path)
print(f"Figure saved to {fig_path}")

plt.close(fig)