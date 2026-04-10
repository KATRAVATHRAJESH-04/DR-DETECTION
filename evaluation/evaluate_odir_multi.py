import os
import sys
import ast
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, multilabel_confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hybrid_model import HybridDRClassifier

# ── CONFIGURATION ──────────────────────────────────────────────────────────
CSV_PATH = "../dataset/raw/archive (3)/full_df.csv"
IMG_DIR = "../dataset/raw/archive (3)/preprocessed_images"
MODEL_PATH = "../models/odir_hybrid_model_v1.pth"
BATCH_SIZE = 16
NUM_CLASSES = 8
DISEASE_NAMES = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']

# ── DATASET CLASS ─────────────────────────────────────────────────────────
class ODIRDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row['filename']
        
        target_str = row['target']
        if isinstance(target_str, str):
            target = ast.literal_eval(target_str)
        else:
            target = target_str
            
        target_tensor = torch.tensor(target, dtype=torch.float32)

        img_path = os.path.join(self.img_dir, filename)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, target_tensor


# ── EVALUATION SCRIPT ───────────────────────────────────────────────────────
def main():
    print("🚀 Initializing ODIR-5K Multi-Disease Evaluation Pipeline...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using Device: {device}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter out missing images
    df['exists'] = df['filename'].apply(lambda f: os.path.exists(os.path.join(IMG_DIR, f)))
    valid_df = df[df['exists']]

    # We only care about the validation set for evaluation
    _, val_df = train_test_split(valid_df, test_size=0.2, random_state=42)
    print(f"📊 Evaluating on {len(val_df)} validation images")

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = ODIRDataset(val_df, IMG_DIR, val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load Model
    print("🧠 Loading Hybrid CNN-ViT Model...")
    model = HybridDRClassifier(num_classes=NUM_CLASSES)
    model.to(device)

    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"✅ Model weights loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠️ Warning: Model weights incompatible. Error: {e}")
    else:
        print(f"⚠️ Warning: Model weights not found at {MODEL_PATH}. Using random weights.")

    model.eval()

    all_preds_probs = []
    all_targets = []

    print(f"🔥 Starting Evaluation loop...")
    with torch.no_grad():
        loop_val = tqdm(val_loader, desc=f"Evaluating")
        for images, targets in loop_val:
            images, targets = images.to(device), targets.to(device)
            # Forward pass
            outputs, _ = model(images, images)
            
            # Since this is multi-label, we apply Sigmoid to get probabilities between 0 and 1
            probs = torch.sigmoid(outputs)
            
            all_preds_probs.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Flatten the batches
    all_preds_probs = np.vstack(all_preds_probs)
    all_targets = np.vstack(all_targets)

    # Convert probabilities to binary predictions using a 0.5 threshold
    threshold = 0.5
    all_preds_binary = (all_preds_probs >= threshold).astype(int)

    # ── METRICS CALCULATION ──
    # For multi-label, "accuracy_score" requires exact match of all labels per instance.
    exact_match_acc = accuracy_score(all_targets, all_preds_binary)
    
    # Calculate Macro and Micro metrics
    macro_precision = precision_score(all_targets, all_preds_binary, average='macro', zero_division=0)
    macro_recall = recall_score(all_targets, all_preds_binary, average='macro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_preds_binary, average='macro', zero_division=0)

    print("\n" + "="*50)
    print("📈 OVERALL METRICS (Multi-label Evaluation)")
    print("="*50)
    print(f"Exact Match Accuracy : {exact_match_acc*100:.2f}%")
    print(f"Macro Precision      : {macro_precision*100:.2f}%")
    print(f"Macro Recall         : {macro_recall*100:.2f}%")
    print(f"Macro F1-Score       : {macro_f1*100:.2f}%")
    
    print("\n📊 PER-DISEASE CLASSIFICATION REPORT")
    print(classification_report(all_targets, all_preds_binary, target_names=DISEASE_NAMES, zero_division=0))

    # ── CONFUSION MATRICES ──
    # Multi-label classification requires calculating a confusion matrix per class
    print("\nGenerating Multi-label Confusion Matrices...")
    mcm = multilabel_confusion_matrix(all_targets, all_preds_binary)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, (matrix, ax) in enumerate(zip(mcm, axes)):
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        ax.set_title(f"{DISEASE_NAMES[i]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    cm_path = "odir_confusion_matrices.png"
    plt.savefig(cm_path, dpi=150)
    print(f"💾 Confusion matrices saved to {cm_path}")

if __name__ == "__main__":
    main()
