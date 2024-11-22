import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class for YOLOv5 Annotations
class YoloV5Dataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Load YOLO label file
        label_path = os.path.join(self.labels_dir, self.label_files[idx])
        boxes = []
        labels = []
        with open(label_path, "r") as file:
            for line in file.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert YOLO format (cx, cy, w, h) to (xmin, ymin, xmax, ymax)
                img_width, img_height = img.size
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id + 1)  # Faster R-CNN expects 1-based class labels

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Define transformations
transform = T.Compose([
    T.ToTensor(),
])

# Paths to your dataset
train_images_dir = r'E:\Vs codes\project\Student Classroom Activity\train\images'
train_labels_dir = r'E:\Vs codes\project\Student Classroom Activity\train\labels'
val_images_dir = r'E:\Vs codes\project\Student Classroom Activity\valid\images'
val_labels_dir = r'E:\Vs codes\project\Student Classroom Activity\valid\labels'

# Create datasets and loaders
train_dataset = YoloV5Dataset(train_images_dir, train_labels_dir, transforms=transform)
val_dataset = YoloV5Dataset(val_images_dir, val_labels_dir, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load Faster R-CNN model pre-trained on COCO
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)

# Define optimizer and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
num_epochs = 20

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}")

# Save the trained model as ONNX
import torch.onnx

# Specify a dummy input to trace the model
dummy_input = torch.randn(1, 3, 640, 640).to(device)  # adjust (1, 3, H, W) as per your input size
onnx_save_path = r"faster_rcnn_model.onnx"

# Export the model to ONNX format
torch.onnx.export(model, dummy_input, onnx_save_path, 
                  export_params=True, 
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'], 
                  output_names=['output'])
print(f"Model saved in ONNX format to {onnx_save_path}")