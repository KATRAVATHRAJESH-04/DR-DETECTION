import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import gc
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.hybrid_model import HybridDRClassifier
except ImportError as e:
    print(f"Error importing model: {e}")

# ================= Configuration =================
class Config:
    SEED = 42
    BATCH_SIZE = 16  # Keep small to avoid Out Of Memory with ViT
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5
    IMAGE_SIZE = 224
    # Dynamic multi-platform dataset paths
    if os.path.exists(r'C:\Users\ASUS\Downloads\aptos2019-blindness-detection\train.csv'):
        DATA_DIR = r'C:\Users\ASUS\Downloads\aptos2019-blindness-detection\train_images'
        CSV_PATH = r'C:\Users\ASUS\Downloads\aptos2019-blindness-detection\train.csv'
    elif os.path.exists('/content/aptos_dataset/train.csv'):
        # Found downloaded Colab folder
        DATA_DIR = '/content/aptos_dataset/train_images'
        CSV_PATH = '/content/aptos_dataset/train.csv'
    else:
        # Default Kaggle paths
        DATA_DIR = '../input/aptos2019-blindness-detection/train_images'
        CSV_PATH = '../input/aptos2019-blindness-detection/train.csv'
    MODEL_SAVE_PATH = 'dr_hybrid_model.pth'

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ================= Loss Function =================
class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance in DR datasets"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

# ================= Data Loader =================
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Match 'id_code' (or similar column in dataset) to `.png` or `.jpeg`
        # APTOS 2019 uses id_code and .png usually
        self.df['image_path'] = self.df['id_code'].apply(lambda x: os.path.join(self.img_dir, f"{x}.png"))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image_path']
        label = self.df.iloc[idx]['diagnosis']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback for weird data
            image = Image.new('RGB', (Config.IMAGE_SIZE, Config.IMAGE_SIZE))
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# ================= Main Training Loop =================
def train():
    seed_everything(Config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Standard fast GPU transforms for Kaggle training
    def simulate_low_res(pil_img):
        """Downsample then upsample to mimic low-resolution uploads."""
        w, h = pil_img.size
        # Keep at least 16x16 after downsample to avoid degenerate images
        scale = random.uniform(0.35, 0.7)
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))
        small = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        return small.resize((w, h), Image.Resampling.BILINEAR)

    def add_gaussian_noise(pil_img):
        """Add mild Gaussian noise to mimic sensor/compression artifacts."""
        arr = np.array(pil_img).astype(np.float32)
        # sigma tuned for uint8 images; keep noise relatively subtle
        noise = np.random.normal(0.0, 12.0, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    train_transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        # Synthetic low-quality augmentations for robustness
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5))], p=0.25),
        transforms.RandomApply([transforms.Lambda(simulate_low_res)], p=0.25),
        transforms.RandomApply([transforms.Lambda(add_gaussian_noise)], p=0.25),
        # Mild illumination/contrast changes (already present, expanded slightly)
        transforms.ColorJitter(brightness=0.25, contrast=0.25),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if paths exist (Kaggle context)
    if not os.path.exists(Config.CSV_PATH):
        print(f"Warning: {Config.CSV_PATH} not found. Ensure Kaggle inputs are connected.")
        return
        
    dataset = DRDataset(Config.CSV_PATH, Config.DATA_DIR, transform=train_transform)
    
    # 80/20 train val split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = HybridDRClassifier(num_classes=Config.NUM_CLASSES)
    model = model.to(device)
    
    # Freeze ViT partially to save RAM and computation
    for param in model.vit.parameters():
        param.requires_grad = False
    
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')

    print("Starting Training on Kaggle...")
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(images, images)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images, images)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the entire model state dictionary
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"--> Saved best model to {Config.MODEL_SAVE_PATH}")
            
        # Clear out Kaggle memory
        gc.collect()
        torch.cuda.empty_cache()

    print("Training Complete. Model is ready for download.")

if __name__ == '__main__':
    train()
