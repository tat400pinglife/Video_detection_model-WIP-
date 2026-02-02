import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
import os
import warnings
import numpy as np

# Import Architecture
from model_architecture import PRNUBranch

# CONFIG
DATA_FOLDER = "./data/processed_data"
SAVE_PATH = "models/noise_model.pth"
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4 

warnings.filterwarnings("ignore")

class RobustNoiseDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).rglob("*.pt"))
        print(f"Index complete. Found {len(self.files)} samples.")
        
        # Fast Label Scan for Class Balancing
        self.labels = []
        valid_files = []
        for f in self.files:
            if "real" in str(f.parent).lower():
                self.labels.append(0)
                valid_files.append(f)
            elif "fake" in str(f.parent).lower():
                self.labels.append(1)
                valid_files.append(f)
            else:
                pass # Skip unknown folders
        
        self.files = valid_files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            path = self.files[idx]
            data = torch.load(path, weights_only=False)
            
            # Extract PRNU
            if 'prnu' not in data: return None
            
            x = data['prnu'].float()
            
            # Ensure shape [1, 256, 256]
            if x.ndim == 2: x = x.unsqueeze(0)
            if x.ndim == 4: x = x.squeeze(0)
            
            # Nan Guard
            if torch.isnan(x).any(): return None
            
            y = torch.tensor([data['label']], dtype=torch.float32)
            return x, y
        except Exception:
            return None

def drop_corrupt_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None, None
    inputs, labels = zip(*batch)
    return torch.stack(inputs), torch.stack(labels)

def train_noise_expert():
    print(f"Using device: {DEVICE}")
    
    # 1. Setup Data
    dataset = RobustNoiseDataset(DATA_FOLDER)
    if len(dataset) == 0: return

    # 2. Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    
    # 3. Balance Classes (Train Set Only)
    train_indices = train_set.indices
    train_labels = [dataset.labels[i] for i in train_indices]
    
    class_counts = np.bincount(train_labels)
    if len(class_counts) < 2:
        sampler = None
    else:
        weights = 1. / (class_counts + 1e-6)
        samples_weights = [weights[l] for l in train_labels]
        sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    # 4. Loaders
    train_loader = DataLoader(
        train_set, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=NUM_WORKERS,
        collate_fn=drop_corrupt_collate,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=drop_corrupt_collate,
        pin_memory=True
    )
    
    # 5. Initialize Model
    net = PRNUBranch().to(DEVICE)
    head = nn.Linear(32*32*32, 1).to(DEVICE)
    
    optimizer = optim.Adam(list(net.parameters()) + list(head.parameters()), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        net.train(); head.train()
        train_loss = 0
        count = 0
        
        for X, y in train_loader:
            if X is None: continue
            X, y = X.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            feats = net(X)
            pred = head(feats)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            count += 1
            
        avg_train_loss = train_loss / max(count, 1)

        # VALIDATE
        net.eval(); head.eval()
        correct = 0; total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                if X is None: continue
                X, y = X.to(DEVICE), y.to(DEVICE)
                
                feats = net(X)
                pred = (torch.sigmoid(head(feats)) > 0.5).float()
                
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = 100 * correct / (total + 1e-8)
        print(f"Epoch {epoch+1} | Loss: {avg_train_loss:.4f} | Val Acc: {acc:.2f}%")
        
        # Save Best
        if acc >= best_acc and acc > 50.0:
            best_acc = acc
            os.makedirs("models", exist_ok=True)
            
            full_state = {
                'prnu_branch.net.0.weight': None, # Dummy key to trigger logic if needed
            }
            # Add Net weights with prefix
            for k, v in net.state_dict().items():
                full_state[f'net.{k}'] = v
            # Add Head weights (if want to load head later)
            for k, v in head.state_dict().items():
                full_state[f'head.{k}'] = v
                
            torch.save(full_state, SAVE_PATH)
            print(f">> Saved Best Model ({acc:.2f}%)")

if __name__ == "__main__":
    train_noise_expert()