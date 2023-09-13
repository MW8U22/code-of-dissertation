import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
import numpy as np
import os

# 数据路径
data_dir = "C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3444/Data/CQT"

# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据
dataset = datasets.ImageFolder(data_dir, transform=transform)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# 设备配置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model_path = "Whale_model_18_3444_CQT.pth"
model = models.resnet18(pretrained=False)
num_classes = len(os.listdir(data_dir))
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

all_preds = []
all_labels = []

# 在验证集上进行预测
for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

# 计算各项指标的比例
total = sum(cm.ravel())
tn_ratio, fp_ratio, fn_ratio, tp_ratio = tn/total, fp/total, fn/total, tp/total

print(f"TN ratio: {tn_ratio:.2%}")
print(f"FP ratio: {fp_ratio:.2%}")
print(f"FN ratio: {fn_ratio:.2%}")
print(f"TP ratio: {tp_ratio:.2%}")


