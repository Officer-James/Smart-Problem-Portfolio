import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------- 模型定义（与训练代码相同）--------------------
class CBAM(torch.nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction_ratio=16):
        super().__init__()
        self.channel_attention = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Conv2d(channels, channels//reduction_ratio, kernel_size=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels//reduction_ratio, channels, kernel_size=1),
            torch.nn.Sigmoid()
        )
        self.spatial_attention = torch.nn.Sequential(
            torch.nn.Conv2d(2, 1, kernel_size=7, padding=3),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        channel = self.channel_attention(x) * x
        spatial_avg = torch.mean(channel, dim=1, keepdim=True)
        spatial_max, _ = torch.max(channel, dim=1, keepdim=True)
        spatial = torch.cat([spatial_avg, spatial_max], dim=1)
        spatial = self.spatial_attention(spatial) * channel
        return spatial

def initialize_enhanced_model(num_classes=2):
    """初始化带CBAM的ResNet18"""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    
    # 冻结前三个layer的参数
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # 在layer4后添加CBAM
    model.layer4.add_module("cbam", CBAM(512))

    # 修改全连接层
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(model.fc.in_features, num_classes))
    return model.to(device)

# -------------------- 加载训练好的模型 --------------------
def load_trained_model(model_path):
    """加载训练好的模型"""
    model = initialize_enhanced_model()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设置为评估模式
    return model

# -------------------- 预测函数 --------------------
def preprocess_image(image_path):
    """预处理图像，与验证集相同"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # 添加batch维度

def predict_image(model, image_tensor):
    """进行预测并返回结果"""
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.softmax(output, dim=1)[0] * 100
        predicted_class = torch.argmax(output).item()
    return predicted_class, probabilities.cpu().numpy()

def visualize_prediction(image_path, predicted_class, probabilities):
    """可视化预测结果"""
    image = Image.open(image_path)
    plt.imshow(image)
    plt.title(f"预测结果: {'错题' if predicted_class == 1 else '非错题'}\n"
              f"非错题概率: {probabilities[0]:.1f}% | 错题概率: {probabilities[1]:.1f}%")
    plt.axis('off')
    plt.show()

# -------------------- 主函数 --------------------
def main():
    # 1. 加载训练好的模型
    model = load_trained_model('best_model.pth')
    print("模型加载成功，准备进行预测...")
    
    # 2. 预测单张图片
    image_path = "test/3.jpg"  # 替换为您的测试图片路径
    input_tensor = preprocess_image(image_path)
    
    # 3. 进行预测
    predicted_class, probabilities = predict_image(model, input_tensor)
    
    # 4. 显示结果
    print(f"预测结果: {'错题' if predicted_class == 1 else '非错题'}")
    print(f"非错题概率: {probabilities[0]:.1f}%")
    print(f"错题概率: {probabilities[1]:.1f}%")
    
    # 5. 可视化结果（可选）
    visualize_prediction(image_path, predicted_class, probabilities)

if __name__ == "__main__":
    main()