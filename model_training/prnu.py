import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import os

from model_architecture import PRNUBranch

class NoiseDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        root_path = Path(root_dir)
        
        self.files = list(root_path.rglob("*.pt"))
        
        if len(self.files) == 0:
            print(f"WARNING: No .pt files found in {root_dir}")
        else:
            random.shuffle(self.files)
            print(f">> Found {len(self.files)} samples for Noise training.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            
            # Extract PRNU (Noise) Tensor
            # Shape in file: [1, 1, 256, 256] or [1, 256, 256]
            prnu_map = data['prnu']
            
            # Ensure shape is [1, 256, 256]
            if prnu_map.ndim == 4:
                prnu_map = prnu_map.squeeze(0)
            elif prnu_map.ndim == 2:
                prnu_map = prnu_map.unsqueeze(0)
            
            # Label
            label = torch.tensor([data['label']], dtype=torch.float32)
            
            # Safety Check: If map is NaN, replace with zeros
            if torch.isnan(prnu_map).any():
                prnu_map = torch.zeros_like(prnu_map)
                
            return prnu_map, label
            
        except Exception as e:
            print(f"Error loading {self.files[idx].name}: {e}")
            return torch.zeros((1, 256, 256)), torch.tensor([0.0])


def train_noise_expert():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"found device:{device}")
    
    # 1. Setup Data
    DATA_ROOT = "./data/processed_data"
    dataset = NoiseDataset(DATA_ROOT)
    
    if len(dataset) == 0:
        print("CRITICAL ERROR: No data found. Please run make_tensors.py first.")
        return
    
    # Batch Loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize Model
    # A. The ConvNet Feature Extractor
    net = PRNUBranch().to(device)
    # B. The Classifier Head (Linear layer)
    head = nn.Linear(32*32*32, 1).to(device)
    
    # Optimizer targets BOTH parts
    optimizer = optim.Adam(list(net.parameters()) + list(head.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Training...")
    
    epochs = 20
    for epoch in range(1, epochs+1):
        total_loss = 0
        correct = 0
        total = 0
        
        net.train()
        head.train()
        
        for prnu_maps, labels in loader:
            prnu_maps = prnu_maps.to(device) # (B, 1, 256, 256)
            labels = labels.to(device)       # (B, 1)
            
            # Skip empty batches
            if prnu_maps.size(0) < 2: continue
            
            optimizer.zero_grad()
            
            features = net(prnu_maps)
            logits = head(features)
            
            loss = criterion(logits, labels)
            
            if torch.isnan(loss):
                print("!! Warning: Loss is NaN. Skipping batch.")
                continue

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = total_loss / len(loader)
        acc = 100 * correct / (total + 1e-8)
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")
        
        if acc > 99.0:
            print(">> Early Stopping: Model has converged.")
            break

    # Save
    os.makedirs("./models", exist_ok=True)
    # save the WHOLE state (Net + Head)
    torch.save({
        'net': net.state_dict(),
        'head': head.state_dict()
    }, "./models/noise_model.pth")
    
    print("\nTraining Complete.")
    print(">> 'noise_model.pth' saved.")

if __name__ == "__main__":
    train_noise_expert()