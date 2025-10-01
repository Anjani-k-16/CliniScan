# %%
!pip install pydicom
import os
import cv2
import pandas as pd
import pydicom  

def dicom_to_png(dicom_path, png_path):
    """Convert a single DICOM file to PNG format."""
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        cv2.imwrite(png_path, img)
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

def convert_dataset(dicom_dir, out_dir, csv_annotations):

    ## converting dicom in csv to png
    os.makedirs(out_dir, exist_ok=True)
    
    annots = pd.read_csv(csv_annotations)

    for i, row in annots.iterrows():
        dicom_file = os.path.join(dicom_dir, row['image_id'] + ".dicom")
        png_file = os.path.join(out_dir, row['image_id'] + ".png")

        if os.path.exists(dicom_file):
            dicom_to_png(dicom_file, png_file)
        else:
            print(f"File not found: {dicom_file}")

# %%
from torch.utils.data import Dataset
from PIL import Image
import torch
class CSRClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):  
        return len(self.image_paths)

    def __getitem__(self, idx):  
        image_path = self.image_paths[idx] 
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = torch.tensor(label, dtype=torch.float32) 
        return image, label



# %%
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


# %%
%pip install "numpy<2.0" --force-reinstall

import torch.nn as nn
import torchvision.models as models

def build_resnet(num_classes):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
        nn.Sigmoid()
    )
    return model

# %%
pip install torch==2.8.0 torchvision --upgrade


# %%

# %pip install torch torchvision

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

train_dir = r"D:\chest_xray\train"
val_dir   = r"D:\chest_xray\val"
test_dir  = r"D:\chest_xray\test"


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


full_train_dataset = datasets.ImageFolder(train_dir, transform=transform)
full_val_dataset   = datasets.ImageFolder(val_dir, transform=transform)
full_test_dataset  = datasets.ImageFolder(test_dir, transform=transform)

max_images = 200  
train_dataset = Subset(full_train_dataset, range(min(len(full_train_dataset), max_images)))
val_dataset   = Subset(full_val_dataset, range(min(len(full_val_dataset), max_images)))
test_dataset  = Subset(full_test_dataset, range(min(len(full_test_dataset), max_images)))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Classes:", full_train_dataset.classes)  


num_classes = len(full_train_dataset.classes)  
model = models.resnet18(weights=None) 

in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, num_classes)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=3):
    for epoch in range(epochs):
       
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return model

model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=30)


model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f" Test Accuracy: {100 * correct / total:.2f}%")



# %%
%pip install ultralytics
from ultralytics import YOLO
import torch, os

print("Torch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))

dataset_root = r"D:\chest_xray"
train_dir = os.path.join(dataset_root, "train")
val_dir   = os.path.join(dataset_root, "val")
test_dir  = os.path.join(dataset_root, "test") 

model = YOLO("yolov8n-cls.pt")

model.train(
    data=dataset_root,
    epochs=20,
    imgsz=224,
    batch=32,
    name="classification_model_fixed"
)

metrics = model.val(data=dataset_root, split="test", imgsz=224, batch=32)
print(metrics)


# %%

# %pip install grad-cam 

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import warnings
import os

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = resnet18(weights=None)  
from torchvision.models import ResNet18_Weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.eval()
model.to(device)


target_layer = model.layer4[-1]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def run_gradcam(image_path):
    
    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)
    rgb_img = np.array(img.resize((224, 224))) / 255.0  # For visualization
    
  
    print("Running Grad-CAM on:", image_path)
    print("Original image shape:", np.array(img).shape)
    print("Input tensor shape:", input_tensor.shape)
    
   
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    
  
    visualization = show_cam_on_image(rgb_img.astype(np.float32), grayscale_cam, use_rgb=True)
    
    return rgb_img, visualization


sample_images = [
    r"D:\chest_xray\val\PNEUMONIA\person1946_bacteria_4874.jpeg",
    r"D:\chest_xray\val\NORMAL\NORMAL2-IM-1427-0001.jpeg"
]


num_images = len(sample_images)
plt.figure(figsize=(10, 5*num_images)) 

for i, img_path in enumerate(sample_images):
    if not os.path.exists(img_path):
        print("Image not found:", img_path)
        continue
    
    orig, heatmap = run_gradcam(img_path)
    
    
    plt.subplot(num_images, 2, 2*i+1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis("off")
    
    
    plt.subplot(num_images, 2, 2*i+2)
    plt.imshow(heatmap)
    plt.title("Grad-CAM")
    plt.axis("off")
    
plt.tight_layout()
plt.show()


save_dir = r"D:\gradcam_outputs"

os.makedirs(save_dir, exist_ok=True)

for img_path in sample_images:
    if not os.path.exists(img_path):
        continue
    _, heatmap = run_gradcam(img_path)
    filename = os.path.basename(img_path).replace(".jpeg", "_gradcam.jpeg")
    save_path = os.path.join(save_dir, filename)
    Image.fromarray(heatmap).save(save_path)
    print("Saved Grad-CAM image:", save_path)


# %%
%matplotlib inline
import glob

image_folder = r"D:\gradcam_outputs"
image_paths = glob.glob(image_folder + "/*.jpeg")

plt.figure(figsize=(12, 6))
for i, path in enumerate(image_paths):
    img = Image.open(path)
    plt.subplot(1, len(image_paths), i+1)
    plt.imshow(img)
    plt.axis("off")
plt.title(f"Image {i+1}")
plt.show()



# %%
streamlit run app.py


