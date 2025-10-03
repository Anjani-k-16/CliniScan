import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import os
import random 
from typing import Dict, List


train_dir = r"D:\chest_xray\chest_xray\train"
val_dir   = r"D:\chest_xray\chest_xray\test"



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


train_data_full = datasets.ImageFolder(train_dir, transform=train_transform)
val_data = datasets.ImageFolder(val_dir, transform=val_transform)

def undersample_data(full_dataset: datasets.ImageFolder) -> Subset:
    """Performs random undersampling to balance classes in the training set."""
    targets = full_dataset.targets
    class_to_idx = full_dataset.class_to_idx

  
    class_indices: Dict[int, List[int]] = {i: [] for i in range(len(class_to_idx))}
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)

   
    minority_size = min(len(indices) for indices in class_indices.values())
    
    print("-" * 40)
    print("Resampling Process:")
    print(f"Classes: {full_dataset.classes}")
    print(f"Original counts: {[len(indices) for indices in class_indices.values()]}")
    print(f"Balancing all classes to size: {minority_size}")
    print("-" * 40)

  
    balanced_indices = []
    for class_idx, indices in class_indices.items():
       
        balanced_indices.extend(random.sample(indices, minority_size))

   
    return Subset(full_dataset, balanced_indices)


train_data_balanced = undersample_data(train_data_full)


train_loader = DataLoader(train_data_balanced, batch_size=32, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=32)



model = models.resnet18(weights='DEFAULT') 

for param in model.parameters():
    param.requires_grad = False


model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 10 
for epoch in range(epochs):
   
    model.train()
    running_loss = 0
    correct, total = 0, 0
    
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
    val_loss, correct, total = 0, 0, 0
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

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


torch.save(model.state_dict(), r"C:\Users\Dell\OneDrive\Desktop\Infosys\model.pth") 
print("\nTraining complete. Model saved as 'model.pth'")

