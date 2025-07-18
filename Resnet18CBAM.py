import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------- 1. 数据增强与数据集定义 --------------------
class AdvancedErrorMarkDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # 加载错题图片 (label=1)
        wrong_dir = os.path.join(data_dir, 'wrong')
        for img in os.listdir(wrong_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(wrong_dir, img))
                self.labels.append(1)

        # 加载正确图片 (label=0)
        right_dir = os.path.join(data_dir, 'right')
        for img in os.listdir(right_dir):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(right_dir, img))
                self.labels.append(0)

        print(f"数据集统计: 共 {len(self.image_paths)} 张图片")
        print(f"错题: {sum(self.labels)}, 非错题: {len(self.labels)-sum(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_advanced_transforms():
    """创建增强的数据转换"""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, shear=10),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return {'train': train_transform, 'val': val_transform, 'test': val_transform}

def create_dataloaders(data_dir, batch_size=32):
    """创建数据加载器"""
    transforms_dict = create_advanced_transforms()
    full_dataset = AdvancedErrorMarkDataset(data_dir, transform=transforms_dict['val'])

    # 分层划分数据集
    train_idx, test_idx = train_test_split(
        range(len(full_dataset)),
        test_size=0.2,
        stratify=full_dataset.labels,
        random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=0.125,  # 0.1/0.8=0.125
        stratify=[full_dataset.labels[i] for i in train_idx],
        random_state=42
    )

    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)

    # 应用不同的转换
    train_dataset.dataset.transform = transforms_dict['train']

    # 类别平衡采样
    class_weights = 1. / torch.tensor(np.bincount(full_dataset.labels), dtype=torch.float)
    sample_weights = class_weights[[full_dataset.labels[i] for i in train_idx]]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # 数据加载器
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            # shuffle=True,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    }

    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    return dataloaders

# -------------------- 2. 模型定义 --------------------
class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel = self.channel_attention(x) * x

        # 空间注意力
        spatial_avg = torch.mean(channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(channel, dim=1, keepdim=True)
        spatial = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial = self.spatial_attention(spatial) * channel

        return spatial

def initialize_enhanced_model(num_classes=2):
    """初始化带CBAM的ResNet18"""
    model = models.resnet18(pretrained=True)

    # 冻结前三个layer的参数
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # 在layer4后添加CBAM
    model.layer4.add_module("cbam", CBAM(512))

    # 修改全连接层
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )

    return model.to(device)

# -------------------- 3. 训练与评估 --------------------
def train_model_advanced(model, dataloaders, num_epochs=50):
    """带混合精度和早停的训练"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs*len(dataloaders['train']), eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    patience, no_improve = 1000, 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss, running_correct = 0.0, 0
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(dataloaders['train'].dataset)
        epoch_acc = running_correct.double() / len(dataloaders['train'].dataset)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu())

        # 验证阶段
        val_loss, val_acc = evaluate_phase(model, dataloaders['val'], criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.cpu())

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')

        # 早停与模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # 加载最佳模型
    model.load_state_dict(best_weights)
    return model, history

def evaluate_phase(model, loader, criterion):
    """评估阶段"""
    model.eval()
    running_loss, running_correct = 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_correct.double() / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate_model(model, loader):
    """完整评估"""
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # 分类报告
    print(classification_report(all_labels, all_preds, target_names=['非错题', '错题']))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Wrong', 'Wrong'],
                yticklabels=['Non-Wrong', 'Wrong'])
    plt.xlabel('Predict')
    plt.ylabel('Real')
    plt.title('Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

# -------------------- 4. 可视化与预测 --------------------
def plot_history(history):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_single_image(model, image_path):
    """单张图片预测"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0] * 100

    plt.imshow(image)
    plt.title(f"Prediction: {'Wrong' if torch.argmax(output).item() else 'Non-Wrong'}\n"
              f"Non-Wrong Probability: {prob[0]:.1f}% | Wrong Probability: {prob[1]:.1f}%")
    plt.axis('off')
    plt.show()

# -------------------- 主函数 --------------------
def main():
    # 1. 准备数据
    data_dir = "dataset"  # 修改为你的数据集路径
    dataloaders = create_dataloaders(data_dir, batch_size=256)

    # 2. 初始化模型
    model = initialize_enhanced_model()
    print("模型参数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 3. 训练模型
    model, history = train_model_advanced(model, dataloaders, num_epochs=25)

    # 4. 评估与可视化
    plot_history(history)
    evaluate_model(model, dataloaders['test'])

    # 5. 示例预测
    # predict_single_image(model, "test_image.jpg")

if __name__ == "__main__":
    print(f"使用设备: {device}")
    main()