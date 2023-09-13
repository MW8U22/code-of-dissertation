
import sys
import random
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from torchvision.datasets import ImageFolder


def load_model(model_path, num_classes):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    # 修改处：使用 map_location 将模型加载到 CPU 或 GPU 上
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()
    return model


#def load_model(model_path, num_classes):
    #  model = models.resnet18(pretrained=False)
    #  model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    #  model.load_state_dict(torch.load(model_path))
    # model.eval()
# return model


def predict(model, image_path, class_names, device):
    transform = transforms.Compose([transforms.Resize((100, 100)),
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

class PlantClassifierApp(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Whale Classifier')
        self.setGeometry(200, 200, 800, 400)

        self.initUI()

        model_path = "Whale_model.pth"
        data_dir = r"Data\Train"

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        num_classes = len(os.listdir(data_dir))
        class_names = ImageFolder(data_dir).classes

        self.model = load_model(model_path, num_classes).to(device)

    def initUI(self):
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(300, 300)
        left_layout.addWidget(self.image_label)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)
        left_layout.addWidget(self.load_button)

        self.classify_button = QPushButton('Classify', self)
        self.classify_button.clicked.connect(self.classify_image)
        left_layout.addWidget(self.classify_button)

        self.prediction_label = QLabel(self)
        left_layout.addWidget(self.prediction_label)

        layout.addLayout(left_layout)

        self.result_labels = []
        for _ in range(6):
            result_label = QLabel(self)
            result_label.setFixedSize(150, 150)
            right_layout.addWidget(result_label)
            self.result_labels.append(result_label)

        layout.addLayout(right_layout)

        self.setLayout(layout)

    def load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.xpm *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.image_path = file_name
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height()))

    def classify_image(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        data_dir = r"data\train"
        class_names = ImageFolder(data_dir).classes
        prediction, confidence = predict(self.model, self.image_path, class_names, device)
        self.prediction_label.setText(f"Predicted class: {prediction}, Confidence: {confidence:.4f}")
        print(f"Predicted class: {prediction}, Confidence: {confidence:.4f}")

        folder_path = os.path.join(data_dir, prediction)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]

        for i in range(6):
            random_image = random.choice(image_files)
            random_image_path = os.path.join(folder_path, random_image)
            pixmap = QPixmap(random_image_path)
            self.result_labels[i].setPixmap(
                pixmap.scaled(self.result_labels[i].width(), self.result_labels[i].height()))
if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = PlantClassifierApp()
    window.show()

    sys.exit(app.exec_())
