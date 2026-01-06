import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

from model_architecture import MoE_Investigator

class ComplexDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        random.shuffle(self.files)
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)
        
        # Prepare inputs for all experts
        rgb = data['rgb'] # (32, 3, 256, 256)
        
        mid_idx = 16
        rgb_mid = rgb[mid_idx] # (3, 256, 256)
        
        prnu = data['prnu'] # (32, 1, 256, 256)
        prnu_var = prnu.var(dim=0)
        if prnu_var.ndim == 2: prnu_var = prnu_var.unsqueeze(0)
            
        label = torch.tensor([data['label']], dtype=torch.float32)
        
        return rgb_mid, rgb, prnu_var, label

def train_investigator():
    device = torch.device("cpu") # or cuda
    
    # 1. Setup Data
    dataset = ComplexDataset("./data/frames")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Setup MoE Model
    print("Building The Investigator...")
    model = MoE_Investigator(
        temp_path="temporal_model.pth",
        art_path="unet_artifact_hunter.pth",
        noise_path="poc_model_256.pth"
    ).to(device)
    
    # 3. Optimizer
    # Notice we filter parameters to ONLY train the Router (and the small noise head)
    # The big experts are ignored/frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=0.001)
    criterion = nn.BCELoss() # Binary Cross Entropy
    
    print("Training the Router logic...")
    
    for epoch in range(1, 21): # this needs more
        total_loss = 0
        model.train()
        
        for rgb_mid, rgb_seq, prnu_var, label in loader:
            rgb_mid = rgb_mid.to(device)
            rgb_seq = rgb_seq.to(device)
            prnu_var = prnu_var.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            verdict, weights = model(rgb_mid, rgb_seq, prnu_var)
            
            loss = criterion(verdict, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        # print(f"  > Sample Weights: Temp={weights[0][0]:.2f}, Art={weights[0][1]:.2f}, Noise={weights[0][2]:.2f}")

    torch.save(model.router.state_dict(), "router_weights.pth")
    print("Router Trained and Saved.")

if __name__ == "__main__":
    train_investigator()