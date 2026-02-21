import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================
# Config
# ==========================
DATA_DIR = "data/test/"
MODEL_PATH = "setA.pth"   # make sure this exists
BATCH_SIZE = 32
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# Transforms
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================
# Dataset
# ==========================
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = dataset.classes
print("Classes:", class_names)
print("Total test samples:", len(dataset))

# ==========================
# Load Model
# ==========================
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)
model.eval()

print("Model Loaded Successfully!")

# ==========================
# Evaluation
# ==========================
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==========================
# Overall Metrics
# ==========================
overall_acc = accuracy_score(all_labels, all_preds)
macro_f1 = f1_score(all_labels, all_preds, average='macro')

print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
print(f"Macro F1 Score: {macro_f1:.4f}")

# ==========================
# Confusion Matrix
# ==========================
cm = confusion_matrix(all_labels, all_preds)

# ==========================
# Class-wise Accuracy
# ==========================
print("\nClass-wise Accuracy:")
for i, class_name in enumerate(class_names):
    correct = cm[i, i]
    total = cm[i].sum()
    acc = 100 * correct / total if total > 0 else 0
    print(f"{class_name}: {acc:.2f}%")

# ==========================
# Class 5 Accuracy
# ==========================
class_index = 5
correct_5 = cm[class_index, class_index]
total_5 = cm[class_index].sum()
class_5_acc = 100 * correct_5 / total_5 if total_5 > 0 else 0

print(f"\nClass 5 Accuracy: {class_5_acc:.2f}%")

# ==========================
# Classification Report
# ==========================
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ==========================
# Plot Confusion Matrix
# ==========================
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=class_names,
            yticklabels=class_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

# save instead of show (safe for Docker/headless)
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# ==========================
# Predict Specific Image
# ==========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    print("\nSingle Image Prediction")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {class_names[pred.item()]}")
    print(f"Confidence: {confidence.item()*100:.2f}%")


# Predict fixed image
predict_image("data/test/5/340.png")