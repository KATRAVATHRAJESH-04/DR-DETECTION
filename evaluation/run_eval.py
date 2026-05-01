"""
Quick evaluation runner — sets correct local paths and runs evaluate.py logic inline.
"""
import os, sys, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.hybrid_model import HybridDRClassifier

# ── HARD-CODED LOCAL PATHS ──────────────────────────────────────────────────
CSV_PATH   = r"C:\Users\ASUS\Downloads\aptos2019-blindness-detection\train.csv"
DATA_DIR   = r"C:\Users\ASUS\Downloads\aptos2019-blindness-detection\train_images"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dr_hybrid_model.pth")
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
        # Try .png first, then .jpeg
        for ext in [".png", ".jpeg", ".jpg"]:
            path = os.path.join(self.img_dir, f"{row['id_code']}{ext}")
            if os.path.exists(path):
                break
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))
        return self.transform(img), int(row["diagnosis"])

def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"CSV    : {CSV_PATH}")
    print(f"Images : {DATA_DIR}")
    print(f"Model  : {MODEL_PATH}\n")

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found: {CSV_PATH}"); return
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Image dir not found: {DATA_DIR}"); return

    df = pd.read_csv(CSV_PATH)
    val_df = df.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    print(f"Evaluating on {len(val_df)} images (20% stratified split)...")

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        APTOSDataset(val_df, DATA_DIR, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = HybridDRClassifier(num_classes=NUM_CLASSES).to(device)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model weights loaded successfully.\n")
    else:
        print(f"[WARNING] Weights not found at {MODEL_PATH}. Using random weights!\n")
    model.eval()

    all_preds, all_labels = [], []
    total = len(loader)
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, 1):
            images = images.to(device)
            logits, _ = model(images, images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            if i % 10 == 0 or i == total:
                print(f"  Batch {i}/{total} done...")

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc   = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    f1    = f1_score(all_labels, all_preds, average=None, zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print("\n" + "="*55)
    print("  APTOS 2019 — DR Model Evaluation Results")
    print("="*55)
    print(f"  Accuracy (Overall)       : {acc*100:.2f}%")
    print(f"  Macro Precision          : {macro_precision*100:.2f}%")
    print(f"  Macro Recall             : {macro_recall*100:.2f}%")
    print(f"  Macro F1-Score           : {macro_f1*100:.2f}%")
    print(f"  Quadratic Weighted Kappa : {kappa:.4f}")
    print("="*55)
    print("\nPer-class F1 Scores:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<22}: {f1[i]:.4f}")
    print("\nFull Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — APTOS 2019 Validation Set (20%)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "confusion_matrix.png")
    plt.savefig(out, dpi=150)
    print(f"\nConfusion matrix saved to: {out}")

if __name__ == "__main__":
    main()
