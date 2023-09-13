
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


model_path = "Whale_model_3376_pcen.pth"
image_path = r"C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索-3376\data\pcen_3376\hw\53A08A45_0_pcen.png"
data_dir = r"data\pcen_3376"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = len(os.listdir(data_dir))
# print(num_classes)
class_names = ImageFolder(data_dir).classes

model = load_model(model_path, num_classes).to(device)
prediction, confidence = predict(model, image_path, class_names, device)

print(f"Predicted class: {prediction}, Confidence: {confidence:.4f}")







from torch.utils.data import DataLoader

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
test_dataset = ImageFolder(r"C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索-3376\data\pcen_3376", transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

accuracy = test_model(model, test_dataloader, device)
print(f"Model accuracy: {accuracy:.4f}")