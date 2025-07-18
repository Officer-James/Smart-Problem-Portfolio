import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import copy

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ErrorMarkDataset(Dataset):
    """自定义错题标记数据集"""
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 数据集目录，应包含 'wrong' 和 'right' 两个子目录
        :param transform: 数据增强转换
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 收集所有图像路径和标签
        self.image_paths = []
        self.labels = []
        
        # 错题图像 (标签为1)
        wrong_dir = os.path.join(data_dir, 'wrong')
        for img_name in os.listdir(wrong_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(wrong_dir, img_name))
                self.labels.append(1)
        
        # 非错题图像 (标签为0)
        right_dir = os.path.join(data_dir, 'right')
        for img_name in os.listdir(right_dir):
            if img_name.endswith(('.png', '.jpg', '.jpeg')):
                self.image_paths.append(os.path.join(right_dir, img_name))
                self.labels.append(0)
        
        print(f"数据集统计: 共 {len(self.image_paths)} 张图片")
        print(f"错题数量: {sum(self.labels)}, 非错题数量: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # 确保为RGB格式
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_datasets(data_dir, test_size=0.2, val_size=0.1, batch_size=128):
    """创建训练、验证和测试数据集"""
    # 数据增强和预处理
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # 创建完整数据集
    full_dataset = ErrorMarkDataset(data_dir, transform=data_transforms['val'])
    
    # 划分训练+验证集和测试集
    train_val_idx, test_idx = train_test_split(
        range(len(full_dataset)), 
        test_size=test_size, 
        stratify=full_dataset.labels,
        random_state=42
    )
    
    # 从训练+验证集中划分验证集
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=val_size/(1-test_size), 
        stratify=[full_dataset.labels[i] for i in train_val_idx],
        random_state=42
    )
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # 为训练集应用不同的转换
    train_dataset.dataset.transform = data_transforms['train']
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    return dataloaders

def initialize_model(num_classes=2, feature_extract=True, use_pretrained=True):
    """初始化模型"""
    model_ft = models.resnet18(pretrained=use_pretrained)
    
    if feature_extract:
        # 冻结所有参数
        for param in model_ft.parameters():
            param.requires_grad = False
    
    # 修改最后一层全连接层
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    
    return model_ft

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """训练模型"""
    since = time.time()
    
    # 保存最佳模型权重
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 训练历史记录
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)
        
        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 训练模式
            else:
                model.eval()   # 评估模式
            
            running_loss = 0.0
            running_corrects = 0
            
            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # 反向传播 + 优化 (仅训练阶段)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 记录历史
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().numpy())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.cpu().numpy())
            
            # 深拷贝模型 (如果是验证阶段且性能更好)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'训练完成于 {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳验证准确率: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    
    # 保存模型
    torch.save(model.state_dict(), 'error_mark_detector1.pth')
    print('模型已保存为 error_mark_detector1.pth')
    
    return model, history

def evaluate_model(model, dataloader):
    """评估模型在测试集上的性能"""
    model.eval()
    
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 分类报告
    print("分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['非错题', '错题']))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['非错题', '错题'], 
                yticklabels=['非错题', '错题'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # 计算准确率
    accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
    print(f"测试集准确率: {accuracy:.4f}")

def plot_training_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='训练损失')
    plt.plot(history['val_loss'], label='验证损失')
    plt.title('损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='训练准确率')
    plt.plot(history['val_acc'], label='验证准确率')
    plt.title('准确率曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def predict_image(model, image_path, class_names=['非错题', '错题']):
    """预测单张图像"""
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(input_batch)
    
    # 获取预测结果
    _, predicted_idx = torch.max(output, 1)
    probability = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    # 显示结果
    plt.imshow(image)
    plt.title(f'预测: {class_names[predicted_idx.item()]}\n'
              f'非错题概率: {probability[0]:.2f}% | 错题概率: {probability[1]:.2f}%')
    plt.axis('off')
    plt.show()
    
    return predicted_idx.item(), probability.cpu().numpy()

def main():
    # 1. 创建数据集和数据加载器
    data_dir = "dataset"  # 替换为您的数据集路径
    dataloaders = create_datasets(data_dir, test_size=0.2, val_size=0.1, batch_size=64)
    
    # 2. 初始化模型
    model = initialize_model(num_classes=2)
    model = model.to(device)
    
    # 3. 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 只训练最后一层（全连接层）
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    
    # 4. 训练模型
    
    
    num_epochs = 20
    model, history = train_model(model, dataloaders, criterion, optimizer, num_epochs=num_epochs)
    
    # 5. 绘制训练历史
    plot_training_history(history)
    
    # 6. 在测试集上评估模型
    evaluate_model(model, dataloaders['test'])
    
    # 7. 示例预测
    # test_image_path = "path/to/test/image.jpg"  # 替换为测试图像路径
    # predict_image(model, test_image_path)

if __name__ == "__main__":
    print(f"使用设备: {device}")
    main()