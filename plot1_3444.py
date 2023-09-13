import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

# Load the model
model_path = "C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3444/Whale_model_18_3444_med.pth"
model = models.resnet18()
num_classes = 2 # adjust this according to your data
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

# Data transformations
data_transforms = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
data_dir = "C:/Users/王民舟/Desktop/final project/鲸鱼检索/鲸鱼检索-3444/Data/med_3444"
full_dataset = ImageFolder(data_dir, transform=data_transforms)

# Split the dataset
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

accuracies = {'Train': [], 'Validation': [], 'Test': []}

for phase, loader in [('Train', train_loader), ('Validation', val_loader), ('Test', test_loader)]:
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracies[phase].append(100 * correct / total)

# Plotting the accuracies
plt.figure()
plt.plot(['Train', 'Validation', 'Test'], [accuracies['Train'][0], accuracies['Validation'][0], accuracies['Test'][0]], marker='o')
plt.ylabel('Accuracy (%)')
plt.xlabel('Data Type')
plt.title('Accuracy of the network on different datasets')
plt.ylim(0, 100)  # Assuming accuracy is in percentage
plt.grid(True)
plt.show()
