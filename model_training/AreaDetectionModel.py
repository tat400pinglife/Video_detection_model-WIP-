import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import random
import numpy as np
from model_architecture import ArtifactSegmentor
import matplotlib.pyplot as plt

# CONFIG
DATA_PATH = "./data/processed_data"
SAVE_PATH = "./models/artifact_model.pth"
BATCH_SIZE = 16
LR = 0.0001
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# class SupervisedArtifactDataset(Dataset):
#     def __init__(self, root_dir):
#         self.files = list(Path(root_dir).rglob("*.pt"))
#         self.clean_files = []
        
#         # Filter and Assign Labels
#         print("Indexing Dataset...")
#         for f in self.files:
#             # We assume folder structure .../real/video.pt or .../fake/video.pt
#             if "fake" in str(f).lower():
#                 self.clean_files.append((f, 1.0)) # Label 1 = Fake
#             elif "real" in str(f).lower():
#                 self.clean_files.append((f, 0.0)) # Label 0 = Real
        
#         random.shuffle(self.clean_files)
#         print(f"Found {len(self.clean_files)} labeled samples.")

#     def __len__(self): return len(self.clean_files)

#     def __getitem__(self, idx):
#         path, label = self.clean_files[idx]
#         try:
#             data = torch.load(path, weights_only=False)
            
#             # Try to get batch or single frame
#             if 'rgb_batch' in data:
#                 frames = data['rgb_batch']
#                 # Pick random frame to train on
#                 ridx = random.randint(0, frames.shape[0]-1)
#                 img = frames[ridx]
#             else:
#                 img = data['rgb_mid']
            
#             # Target: If Real (0.0), Mask is all Zeros.
#             #         If Fake (1.0), Mask is all Ones (Simple Supervised Baseline)
#             #         (Ideally we would have ground-truth masks, but global labels work for classification)
#             target = torch.full((1, 256, 256), label, dtype=torch.float32)
            
#             return img.float(), target
            
#         except:
#             return torch.zeros(3,256,256), torch.zeros(1,256,256)

class SupervisedArtifactDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        print(f"Artifact Dataset: Found {len(self.files)} samples.")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            # 1. LOAD COMPRESSED RGB [32, 3, 256, 256] (Uint8)
            if 'rgb_batch' in data:
                # Pick 1 random frame from the batch
                ridx = random.randint(0, data['rgb_batch'].shape[0]-1)
                img_uint8 = data['rgb_batch'][ridx] # Shape: [3, 256, 256]
                
                # 2. DECOMPRESS: Uint8 (0-255) -> Float32 (0.0-1.0)
                img = img_uint8.float() / 255.0
            else:
                # Fallback for old files (if any)
                img = data['rgb_mid'].float()

            # 3. Create Label/Target
            label = float(data['label']) # 1.0 or 0.0
            
            # Create a segmentation mask (1.0 for Fake, 0.0 for Real)
            # This is a "Weakly Supervised" approach
            target = torch.full((1, 256, 256), label, dtype=torch.float32)
            
            return img, target
            
        except Exception as e:
            # Return dummy if file is corrupted
            return torch.zeros(3, 256, 256), torch.zeros(1, 256, 256)

def train_supervised():
    dataset = SupervisedArtifactDataset(DATA_PATH)
    
    # Split
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = ArtifactSegmentor().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss() # Pixel-wise loss
    
    best_val_loss = 100.0
    print(f"--- Starting Supervised Training on {len(train_ds)} samples ---")
    patience = 3
    trigger_times = 0
    
    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    training_acc_data = []
    

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if x.sum() == 0: continue
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            # training acc
            train_acc = (torch.sigmoid(pred) > 0.5).float()
            
            
        avg_train = train_loss / len(train_loader)
        train_loss_data.append(avg_train)
        training_acc_data.append(int(train_acc.sum().item()) / (len(train_loader.dataset) + 1e-8) * 100)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                if x.sum() == 0: continue
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                # validation acc
                val_acc = (torch.sigmoid(pred) > 0.5).float()
        
        
        avg_val = val_loss / len(val_loader)
        val_acc_data.append(int(val_acc.sum().item()) / (len(val_loader.dataset) + 1e-8) * 100)
        print(f"Epoch {epoch+1} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH) 
                print("Model saved.")
                trigger_times = 0            
            
        else:
            trigger_times += 1
            print(f"No improvement for {trigger_times} epochs.")
            
            # 3. Stop if we haven't improved in a while
            if trigger_times >= patience:
                print("Early Stopping!")
                break
    # plot loss and val_acc
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("artifact_model_training_loss.png")
    
    plt.figure(figsize=(10,5))
    plt.plot(training_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("artifact_model_training_accuracy.png")
    print("Done.")


if __name__ == "__main__":
    train_supervised()