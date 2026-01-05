import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# IMPORT THE MODEL
from model_architecture import TinyDeepfakeDetector

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        if not self.files:
            print(f"Warning: No .pt files found in {root_dir}")
        random.shuffle(self.files) 
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)
        mid_idx = 16 
        
        rgb = data['rgb'][mid_idx]
        diff = data['diff'][mid_idx]
        fft = data['fft'][mid_idx]
        prnu_var = data['prnu'].var(dim=0)
        prnu_mean = data['prnu'].mean(dim=0)
        label = torch.tensor(data['label'], dtype=torch.float32)
        
        return rgb, diff, fft, prnu_mean, prnu_var, label

def train_classifier():
    dataset = DeepfakeDataset("./data/frames") 
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Initialize from imported class
    model = TinyDeepfakeDetector()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 

    print(f"Training Classifier on {len(dataset)} items...")
    
    for epoch in range(1, 20):
        total_loss = 0
        correct = 0
        total = 0
        model.train()
        
        for rgb, diff, fft, prnu_mean, prnu_var, labels in loader:
            optimizer.zero_grad()
            outputs = model(rgb, diff, fft, prnu_mean, prnu_var).view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += len(labels)
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f} | Acc: {correct/total:.1%}")

    torch.save(model.state_dict(), "poc_model_256.pth")
    print("Classifier Saved.")

if __name__ == "__main__":
    train_classifier()