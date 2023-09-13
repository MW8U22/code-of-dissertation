import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import os

# 数据路径
data_dir = "C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3376/Data/Train"

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
model_path = "C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3376/Whale_model_3376_18_1.pth"
model = models.resnet18(pretrained=False)
num_classes = len(os.listdir(data_dir))
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

all_probs = []
all_labels = []

# 在验证集上进行预测
for inputs, labels in val_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.nn.functional.softmax(outputs, dim=1)  # 获取softmax结果
        all_probs.extend(probs[:, 1].cpu().numpy())  # 提取正例的概率值
        all_labels.extend(labels.cpu().numpy())

# 计算ROC曲线
fpr, tpr, _ = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='CQT ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
