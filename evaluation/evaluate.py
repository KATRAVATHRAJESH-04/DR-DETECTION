import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_model import HybridDRClassifier

# ============================================================
# Configuration — update these paths if your data is elsewhere
# ============================================================
DATA_DIR  = os.environ.get("APTOS_DATA_DIR",  os.path.join(os.path.expanduser("~"), "aptos2019", "train_images"))
CSV_PATH  = os.environ.get("APTOS_CSV_PATH",  os.path.join(os.path.expanduser("~"), "aptos2019", "train.csv"))
MODEL_PATH = "dr_hybrid_model.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 5
SEED = 42

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return self.transform(img), int(row["diagnosis"])


def evaluate():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Validate paths ----
    if not os.path.exists(CSV_PATH):
        print(f"\n[ERROR] CSV not found: {CSV_PATH}")
        print("Set the APTOS_CSV_PATH environment variable to your train.csv location.")
        print("Example (PowerShell):  $env:APTOS_CSV_PATH='C:\\path\\to\\train.csv'")
        return
    if not os.path.exists(DATA_DIR):
        print(f"\n[ERROR] Image folder not found: {DATA_DIR}")
        print("Set the APTOS_DATA_DIR environment variable to your train_images folder.")
        return

    # ---- Data ----
    df = pd.read_csv(CSV_PATH)
    val_df = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    print(f"Evaluating on {len(val_df)} images (20% split)...")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loader = DataLoader(APTOSDataset(val_df, DATA_DIR, val_transform),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ---- Load Model ----
    model = HybridDRClassifier(num_classes=NUM_CLASSES).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"[WARNING] Weights not found at {MODEL_PATH}. Using random weights!")
    model.eval()

    # ---- Inference ----
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits, _ = model(images, images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ---- Metrics ----
    acc   = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    f1    = f1_score(all_labels, all_preds, average=None, zero_division=0)

    print("\n" + "="*50)
    print(f"  Accuracy               : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Quadratic Weighted Kappa: {kappa:.4f}")
    print("="*50)
    print("\nPer-class F1 Scores:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22}: {f1[i]:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — APTOS 2019 Validation Set")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\nConfusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
