import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# =========================
# 1. Configuration
# =========================

SEED = 42
BATCH_SIZE = 16
EPOCHS = 8
IMAGE_SIZE = 224
LEARNING_RATE = 0.001

DATA_DIR = Path(
    "datasets/Tennis Player Actions Dataset for Human Pose Estimation/"
    "Tennis Player Actions Dataset for Human Pose Estimation/images"
)

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("TennisAI - Tennis Action Classification")
print("=" * 50)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# =========================
# 2. Validate dataset
# =========================

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATA_DIR.resolve()}"
    )

base_dataset = datasets.ImageFolder(DATA_DIR)

print("\nClasses:", base_dataset.classes)
print("Total images:", len(base_dataset))

if len(base_dataset.classes) != 4:
    raise ValueError(
        f"Expected 4 classes, found {len(base_dataset.classes)}"
    )

if len(base_dataset) == 0:
    raise ValueError("No images found in dataset.")

class_names = base_dataset.classes
targets = np.array(base_dataset.targets)
indices = np.arange(len(base_dataset))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=SEED,
    stratify=targets
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=SEED,
    stratify=targets[temp_idx]
)

print("\nTrain images:", len(train_idx))
print("Validation images:", len(val_idx))
print("Test images:", len(test_idx))


# =========================
# 3. Image transformations
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=train_transform
)

eval_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=eval_transform
)

train_loader = DataLoader(
    Subset(train_dataset, train_idx.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    Subset(eval_dataset, val_idx.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    Subset(eval_dataset, test_idx.tolist()),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =========================
# 4. Load pretrained model
# =========================

print("\nLoading pretrained ResNet18...")

try:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
except Exception as error:
    print("\nCould not download pretrained weights.")
    print("Reason:", error)
    print("Trying to use an untrained ResNet18 instead.")

    model = models.resnet18(weights=None)

    # If weights are unavailable, train more than the final layer.
    for parameter in model.parameters():
        parameter.requires_grad = True

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    len(class_names)
)

model = model.to(device)

# Freeze the feature extractor when pretrained weights are available.
if "weights" in locals():
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()


# =========================
# 5. Training and validation
# =========================

best_val_accuracy = 0.0
best_model_path = OUTPUT_DIR / "tennis_actions_resnet18.pth"


def evaluate(data_loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total else 0.0
    average_loss = total_loss / total if total else 0.0

    return average_loss, accuracy


for epoch in range(EPOCHS):
    model.train()

    total = 0
    correct = 0
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    train_loss = total_loss / total
    train_accuracy = correct / total

    val_loss, val_accuracy = evaluate(val_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.3f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.3f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": IMAGE_SIZE,
            "architecture": "resnet18"
        }, best_model_path)


# =========================
# 6. Final test
# =========================

print("\nLoading best model for final testing...")

checkpoint = torch.load(
    best_model_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.tolist())

print("\nFinal Test Results")
print("Best validation accuracy:", round(best_val_accuracy, 4))

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("Confusion Matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names)))
))

print("\nSaved model:", best_model_path.resolve())
print("Training completed.")