import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from pathlib import Path
import random
import numpy as np
import os

from model_architecture import ArtifactSegmentor

# CONFIG
DATA_FOLDER = "./data/processed_data" 
SAVE_PATH = "models/unet_artifact_hunter.pth"
BATCH_SIZE = 16
LR = 0.001 # Boosted from 0.0001
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArtifactDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).rglob("*.pt"))
        self.labels = []
        
        # Pre-scan to get labels for balancing
        print(f"Scanning {len(self.files)} files for artifacts...")
        valid_files = []
        for f in self.files:
            try:
                # Lightweight check: Read label without loading full tensor if possible
                # For now, we load it. It's slower but safer.
                data = torch.load(f, weights_only=False)
                lbl = int(data['label'])
                
                # Check if RGB data exists
                if data['rgb_batch'].nelement() > 0:
                    valid_files.append(f)
                    self.labels.append(lbl)
            except:
                continue
        
        self.files = valid_files
        print(f"Training on {len(self.files)} valid samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            rgb_batch = data['rgb_batch'] # [Frames, 3, 256, 256]
            label = torch.tensor([data['label']], dtype=torch.float32)

            # Pick 1 Random Frame
            if rgb_batch.size(0) > 0:
                frame_idx = random.randint(0, rgb_batch.size(0)-1)
                img = rgb_batch[frame_idx]
            else:
                img = torch.zeros(3, 256, 256)
            
            return img.float(), label
        except:
            return torch.zeros(3, 256, 256), torch.tensor([0.0])

def train():
    print(f"--- Training ARTIFACT Expert (Mean Pooling) on {DEVICE} ---")
    
    dataset = ArtifactDataset(DATA_FOLDER)
    if len(dataset) == 0: return

    # keep data balanced
    class_counts = np.bincount(dataset.labels)
    print(f"Data Balance: {class_counts[0]} Real vs {class_counts[1]} Fake")
    
    # Stratified Split (Keep balance in train/val)
    # We'll just use random split but check it
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = ArtifactSegmentor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            mask_logits = model(X) # [B, 1, 256, 256]
            
            # Max pooling on noise = Always Fake.
            # Mean pooling on noise = ~0 (Neutral).
            video_score_logits = mask_logits.mean(dim=(1, 2, 3)).unsqueeze(1)
            
            loss = criterion(video_score_logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                mask_logits = model(X)
                
                # Use same pooling for validation
                score = torch.sigmoid(mask_logits.mean(dim=(1, 2, 3)).unsqueeze(1))
                
                pred = (score > 0.5).float()
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / (total + 1e-8)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        if acc >= best_acc:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"Done. Saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()