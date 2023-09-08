import os
import json

import torch
from torchvision import transforms
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
from PIL import Image

class Dataset(torch.utils.data.Dataset):
    """ Класс для загрузки датасета"""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(os.path.join(data_dir, "images")))
        self.annotation_files = sorted(os.listdir(os.path.join(data_dir, "annot")))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, "images", self.image_files[idx])
        ann_path = os.path.join(self.data_dir, "annot", self.annotation_files[idx])
        
        with open(ann_path, 'r') as f:
            annotation = json.load(f)[0]
        
        label = annotation["name"]
        label_map = {"rectangle": 0, "circle": 1, "triangle": 2, "hexagon": 3} 
        label_idx = label_map[label]
        
        origin = annotation["region"]["origin"]
        size = annotation["region"]["size"]
        
        x1 = int(origin["x"])
        y1 = int(origin["y"])
        width = int(size["width"])
        height = int(size["height"])
        
        bbox = [x1, y1, x1 + width, y1 + height]
        
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        
        return image, (torch.tensor(label_idx, dtype=torch.int64), torch.tensor(bbox, dtype=torch.float32))

class SimpleDetector(nn.Module):
    def __init__(self, num_clusses=4):
        super(SimpleDetector, self).__init__()
        
        self.base_model = resnet18(pretrained=True)
        self.base_model.fc = nn.Sequential()

        self.fc1 = nn.Linear(512, 256)
        self.fc2_class = nn.Linear(256, num_clusses)
        self.fc2_bbox = nn.Linear(256, 4)

    def forward(self, x):
        x = self.base_model(x)
        x = F.relu(self.fc1(x))
        class_out = self.fc2_class(x)
        bbox_out = self.fc2_bbox(x)
        return class_out, bbox_out

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, class_preds, class_targets, bbox_preds, bbox_targets):
        return self.cls_loss(class_preds, class_targets) + \
               self.bbox_loss(bbox_preds, bbox_targets)


def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    loss_history = []

    for epoch in range(num_epochs):
        for batch_idx, (inputs, (labels, bboxes)) in enumerate(dataloader):  
            optimizer.zero_grad()
            
            class_preds, bbox_preds = model(inputs)
            loss = criterion(class_preds, labels, bbox_preds, bboxes)

            loss_history.append(loss.item())

            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item()}")
        scheduler.step()

    torch.save(model.state_dict(), 'simple_detector.pth')

    return loss_history

def test_image(model, image_path, transform):
    model.eval()  # переключаем модель в режим предсказания

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # добавляем дополнительное измерение батча

    with torch.no_grad():
        class_preds, bbox_preds = model(image)

    predicted_class = class_preds.argmax(dim=1).item()
    predicted_bbox = bbox_preds.squeeze().tolist()

    label_map = {0: "rectangle", 1: "circle", 2: "triangle", 3: "hexagon"}
    print("Predicted class:", label_map[predicted_class])
    print("Predicted bbox:", predicted_bbox)


if __name__ == "__main__":
    # Определение трансформаций и загрузчика данных
    transform= transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Dataset("generated_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Инициализация модели, функции потерь и оптимизатора
    model = SimpleDetector()
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    # Обучение модели
    loss_history = train_model(model, dataloader, criterion, optimizer)

    # Сохранение весов модели
    torch.save(model.state_dict(), 'simple_detector.pth')

    # Визуализация истории потерь
    plt.plot(loss_history)
    plt.title('Training Loss over Time')
    plt.xlabel('Batch number')
    plt.ylabel('Loss')
    plt.show()

    # Тестирование модели
    test_image_path = "generated_images/001.png"
    test_image(model, test_image_path, transform)