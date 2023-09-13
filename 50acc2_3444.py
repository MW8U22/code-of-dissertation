import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def test_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# 定义数据转换
transform = transforms.Compose([transforms.Resize((100, 100)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 定义数据集和数据加载器
# 更改路径到验证集
val_dataset_path = r"C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索-3444\Data\Train"  # <-- 请确保这是你的验证集路径
val_dataset = ImageFolder(val_dataset_path, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

model_path = "Whale_model_50_3444_1.pth"
num_classes = len(os.listdir(val_dataset_path))
class_names = ImageFolder(val_dataset_path).classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = load_model(model_path, num_classes).to(device)

accuracy = test_model(model, val_dataloader, device)
print(f"Validation accuracy: {accuracy:.4f}")
