import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        if not isinstance(img_path, str):
            print(f"Invalid image path type: {type(img_path)}")
            print(f"Image path value: {img_path}")

        try:
            img = Image.open(img_path).convert("RGB")
        except AttributeError:
            print(f"Invalid image path: {img_path}")
            raise

        if self.transform is not None:
            img = self.transform(img)

        return img, label

# 定义训练和验证数据转换
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((100,100)),
#         transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize((100,100)),
#         transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 数据集路径
data_dir = "data/train"

# 加载整个数据集
full_dataset = ImageFolder(data_dir, transform=data_transforms["train"])

# 将整个数据集分为训练集和验证集
train_indices, val_indices = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)

# 使用数据集的子集创建训练集和验证集
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# 更新验证集的转换
val_dataset.dataset.transform = data_transforms["val"]

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=True)


image,label=next(iter(train_loader))
image.shape

model = models.resnet50(pretrained=True)  # 使用ResNet-50模型

# 修改最后一层以匹配类别数量


num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 定义损失函数、优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 使用GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

from tqdm import tqdm
num_epochs = 25
best_acc = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"Epoch {epoch}/{num_epochs - 1}")

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
            dataloader = train_loader
            dataset_size = len(train_dataset)
        else:
            model.eval()
            dataloader = val_loader
            dataset_size = len(val_dataset)

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        if phase == "train":
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
        else:
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_acc)
        if phase == "val" and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), "Whale_model_50.pth")
    print()

print(f"Best val Acc: {best_acc:.4f}")

import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(num_epochs), train_losses, label="Train Loss")
plt.plot(range(num_epochs), val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")
# plt.savefig("loss_curve.png")



plt.figure()
plt.plot(range(num_epochs), train_accuracies.numpy(), label="Train acc")
plt.plot(range(num_epochs), val_accuracies.numpy(), label="Validation acc")
plt.xlabel("Epoch")
plt.ylabel("acc")
plt.legend()
plt.title("ACC Curve")
# plt.savefig("ACC_curve.png")


import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from torchvision.datasets import ImageFolder


def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, image_path, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),

        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        confidence, preds = torch.max(outputs.softmax(dim=1), 1)

    return class_names[preds], confidence.item()


model_path = "Whale_model_50.pth"
image_path = r"C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索\data\train\hw\53A08A45_0.jpg"
data_dir = r"data\train"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(os.listdir(data_dir))
# print(num_classes)
class_names = ImageFolder(data_dir).classes

model = load_model(model_path, num_classes).to(device)
prediction, confidence = predict(model, image_path, class_names, device)

print(f"Predicted class: {prediction}, Confidence: {confidence:.4f}")

