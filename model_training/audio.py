import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
import random
import numpy as np

from model_architecture import AudioExpert

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        self.clean_files = []
        self.labels = [] # For sampler
        
        print("Filtering dataset...")
        for f in self.files:
            try:
                # Quick load to check label/content
                data = torch.load(f, weights_only=False)
                if data['audio'].max() > 0.01: # Skip silence
                    self.clean_files.append(f)
                    self.labels.append(int(data['label']))
            except: pass
            
        print(f"Training on {len(self.clean_files)} valid audio files.")

    def __len__(self): return len(self.clean_files)

    def __getitem__(self, idx):
        data = torch.load(self.clean_files[idx], weights_only=False)
        spec = data['audio']
        if spec.ndim == 4: spec = spec.squeeze(0)
        if spec.ndim == 2: spec = spec.unsqueeze(0)
        label = torch.tensor([data['label']], dtype=torch.float32)
        return spec.float(), label

def train():
    dataset = AudioDataset("./data/processed_data")
    if len(dataset) == 0: return

    class_counts = np.bincount(dataset.labels)
    if len(class_counts) < 2: 
        print("Error: Only one class found in data"); return
        
    weights = 1. / class_counts
    samples_weights = [weights[l] for l in dataset.labels]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    loader = DataLoader(dataset, batch_size=16, sampler=sampler)
    
    model = AudioExpert().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low LR
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Balanced Training...")
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for specs, labels in loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            out = model(specs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f} | Acc: {100*correct/total:.2f}%")
        
        if 100*correct/total > 60:
            torch.save(model.state_dict(), "models/audio_model.pth")
            print(">> Saved Model")

if __name__ == "__main__":
    train()