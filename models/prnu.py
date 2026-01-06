import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from model_architecture import PRNUBranch

class NoiseDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        random.shuffle(self.files)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        # Extract PRNU Variance (B, 1, 256, 256)
        prnu = data['prnu']
        prnu_var = prnu.var(dim=0)
        if prnu_var.ndim == 2: prnu_var = prnu_var.unsqueeze(0)
        return prnu_var, torch.tensor(data['label'], dtype=torch.float32)

def train_noise_expert():
    device = torch.device("cpu")
    dataset = NoiseDataset("./data/frames")
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # We combine the feature extractor + a temporary classification head
    backbone = PRNUBranch()
    head = nn.Linear(32*32*32, 1)
    
    optimizer = optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training Noise Expert...")
    for epoch in range(1, 11):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            feats = backbone(x)
            out = head(feats).view(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Inside the training loop
        avg_loss = total_loss/len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        if avg_loss < 0.15:
            print(">> Early Stopping: Model is starting to memorize.")
            break
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
    

    # Save the WHOLE state (Backbone + Head)
    # We save it as a dictionary so MoE can load it cleanly
    torch.save({
        'net.0.weight': backbone.net[0].weight, # Mapping might be tricky, saving full object is safer
        'prnu_branch': backbone.state_dict(),
        'head': head.state_dict()
    }, "noise_expert_full.pth")
    
    # SIMPLER: Just save the backbone for the MoE
    torch.save(backbone.state_dict(), "poc_model_256.pth") 
    print("Noise Expert Saved.")

if __name__ == "__main__":
    train_noise_expert()