import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

from model_architecture import PRNUBranch


class NoiseDataset(Dataset):
    def __init__(self, root_dirs):
        self.files = []
        for d in root_dirs:
            self.files.extend(list(Path(d).rglob("*.pt")))
        
        random.shuffle(self.files)
        print(f">> Found {len(self.files)} samples for Noise training.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            
            # The file contains 'prnu': (1, 1, 256, 256) -> The Variance Map
            # We squeeze the first dim to get (1, 256, 256)
            prnu_map = data['prnu'].squeeze(0)
            
            # Label
            label = torch.tensor([data['label']], dtype=torch.float32)
            
            # If map is NaN, replace with zeros
            if torch.isnan(prnu_map).any():
                prnu_map = torch.zeros_like(prnu_map)
                
            return prnu_map, label
            
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros((1, 256, 256)), torch.tensor([0.0])


def train_noise_expert():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Noise Expert on {device} ---")
    
    # Paths
    train_dirs = ["./data/processed_data/real", "./data/processed_data/fake"]
    dataset = NoiseDataset(train_dirs)
    
    # Batch Loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize Model components
    # 1. The ConvNet Feature Extractor
    net = PRNUBranch().to(device)
    # 2. The Classifier Head (Linear layer)
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
        acc = correct / total if total > 0 else 0
        
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.2%}")
        if avg_loss < 0.15:
            print(">> Early Stopping: Model is performing well.")
            break

    # Save
    # We save the WHOLE state (Net + Head) so the main script can load it intelligently
    torch.save({
        'net': net.state_dict(),
        'head': head.state_dict()
    }, "./models/poc_model_256.pth")
    print("\nTraining Complete.")
    print(">> 'poc_model_256.pth' saved.")

if __name__ == "__main__":
    train_noise_expert()