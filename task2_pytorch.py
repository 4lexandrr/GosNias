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
        """
        Эта функция получает каждый объект на изображении по индексу.
        idx: индекс объекта.
        """
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
    """ Класс нейронной сети """
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
    """ Обучение модели """
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
        
        if epoch == 4:
            torch.save(model.state_dict(), 'intermediate_checkpoint.pth')

    return loss_history

def test_image(model, image_path, transform):
    """ Тестирование модели """
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

def calculate_iou(box1, box2):

    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area

    return iou

def evaluate_model(model, dataloader, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    ious = []
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for inputs, (labels, bboxes) in dataloader:
            class_preds, bbox_preds = model(inputs)
            for i in range(inputs.size(0)):
                predicted_bbox = bbox_preds[i]
                true_bbox = bboxes[i]
                
                iou = calculate_iou(predicted_bbox, true_bbox) # передаем два bbox в функцию
                ious.append(iou)
                    
                predicted_class = class_preds[i].argmax(dim=0).item()  # убедимся, что используем индексированный класс
                true_class = labels[i].item()  # аналогично, используем индексированный label
                
                if iou > 0.5:
                    if predicted_class == true_class:
                        tp += 1
                    else:
                        fp += 1
                else:
                    fn += 1

    max_iou = max(ious)
    min_iou = min(ious)
    avg_iou = sum(ious) / len(ious)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return max_iou, min_iou, avg_iou, precision, recall

if __name__ == "__main__":
    # Определение трансформаций и загрузчика данных
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = Dataset("generated_images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Инициализация модели, функции потерь и оптимизатора
    model = SimpleDetector()
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    max_iou1, min_iou1, avg_iou1, precision1, recall1 = evaluate_model(model, dataloader, 'intermediate_checkpoint.pth')
    max_iou2, min_iou2, avg_iou2, precision2, recall2 = evaluate_model(model, dataloader, 'simple_detector.pth')

    print("Intermediate Checkpoint:")
    print(f"Max IoU: {max_iou1}, Min IoU: {min_iou1}, Avg IoU: {avg_iou1}, Precision: {precision1}, Recall: {recall1}")

    print("Final Checkpoint:")
    print(f"Max IoU: {max_iou2}, Min IoU: {min_iou2}, Avg IoU: {avg_iou2}, Precision: {precision2}, Recall: {recall2}")


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