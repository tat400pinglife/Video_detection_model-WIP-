import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
import numpy as np

from model_architecture import TemporalDetector, TinyDeepfakeDetector

class SequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        if not self.files: print("No files found.")
        random.shuffle(self.files)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = torch.load(path, weights_only=False)
        
        # Get RGB Sequence: (32, 3, 256, 256)
        # Permute to (Time, Channel, H, W) if stored differently
        rgb = data['rgb'] 
        
        # Label: If video is fake (1), all frames are labeled 1
        label_val = data['label']
        labels = torch.full((32, 1), label_val, dtype=torch.float32)
        
        return rgb, labels

def train_temporal():
    device = torch.device("cpu") # or cuda
    dataset = SequenceDataset("./data/frames")
    if len(dataset) == 0: return
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("Loading pretrained Classifier weights...")
    classifier = TinyDeepfakeDetector()
    try:
        classifier.load_state_dict(torch.load("poc_model_256.pth", map_location=device, weights_only=True))
        # Pass the RGB branch to the Temporal Detector
        pretrained_branch = classifier.rgb_branch
    except FileNotFoundError:
        print("Warning: poc_model_256.pth not found. Training Temporal from scratch (Harder).")
        pretrained_branch = None

    # Initialize Temporal Model with the "eyes" of the classifier
    model = TemporalDetector(pretrained_cnn=pretrained_branch).to(device)
    
    # Lower learning rate slightly for LSTM
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Temporal Training...")
    
    for epoch in range(1, 16):
        total_loss = 0
        model.train()
        
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(frames) # (B, 32, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
    torch.save(model.state_dict(), "temporal_model.pth")
    print("Temporal Model Saved.")

if __name__ == "__main__":
    train_temporal()