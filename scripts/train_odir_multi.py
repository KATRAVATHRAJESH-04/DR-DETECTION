import os
import sys
import ast
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.hybrid_model import HybridDRClassifier

# ── CONFIGURATION ──────────────────────────────────────────────────────────
CSV_PATH = "../dataset/raw/archive (3)/full_df.csv"
IMG_DIR = "../dataset/raw/archive (3)/preprocessed_images"
MODEL_SAVE_PATH = "../models/odir_hybrid_model_v1.pth"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-4

# ODIR Classes: Normal(N), Diabetes(D), Glaucoma(G), Cataract(C), AMD(A), Hypertension(H), Myopia(M), Other(O)
DISEASE_NAMES = ['Normal', 'Diabetic Retinopathy', 'Glaucoma', 'Cataract', 'AMD', 'Hypertension', 'Myopia', 'Other']
NUM_CLASSES = 8

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
        
        # Parse target string "[0, 1, 0...]" safely to a list of integers
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
            # Fallback to a black image if missing
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, target_tensor


# ── TRAINING SCRIPT ───────────────────────────────────────────────────────
def main():
    print("🚀 Initializing ODIR-5K Multi-Disease Training Pipeline...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Using Device: {device}")

    # Load dataframe
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing CSV: {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    
    # Filter out missing images just in case
    print("🔍 Validating image files...")
    df['exists'] = df['filename'].apply(lambda f: os.path.exists(os.path.join(IMG_DIR, f)))
    valid_df = df[df['exists']]
    print(f"📊 Found {len(valid_df)} valid ocular images out of {len(df)}")

    # Train/Validation Split (80/20)
    train_df, val_df = train_test_split(valid_df, test_size=0.2, random_state=42)

    # Transforms (Resizing + Normalizing for ResNet/ViT)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_dataset = ODIRDataset(train_df, IMG_DIR, train_transform)
    val_dataset = ODIRDataset(val_df, IMG_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Load Model (Hybrid CNN + ViT)
    print("🧠 Building Hybrid CNN-ViT Model...")
    model = HybridDRClassifier(num_classes=NUM_CLASSES)
    model.to(device)

    # Multi-label objective needs BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')

    print(f"🔥 Starting Training loop for {EPOCHS} epochs...\n")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, targets in loop:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            
            # Forward pass (model expects img_vit and img_cnn; we pass the same image to both)
            outputs, _ = model(images, images)
            loss = criterion(outputs, targets)
            
            # Backprop
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            loop_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
            for images, targets in loop_val:
                images, targets = images.to(device), targets.to(device)
                outputs, _ = model(images, images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                loop_val.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"📈 Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"💾 Improved Validation Loss! Weights saved to {MODEL_SAVE_PATH}")
        print("-" * 60)

    print(f"✅ Training Complete. Best model saved to: {MODEL_SAVE_PATH}")
    print("To use this model in the application, update api/main.py to load this weights file and adjust class mappings!")


if __name__ == "__main__":
    main()
