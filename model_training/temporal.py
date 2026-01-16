import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from model_architecture import TemporalDetector

class SequenceDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        random.shuffle(self.files)
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        rgb = data['rgb'] # (32, 3, 256, 256)
        label = float(data['label'])
        labels = torch.full((32, 1), label, dtype=torch.float32)
        return rgb, labels

def train_temporal_expert():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        # Enable Benchmark Mode (optimizes C++ kernels for your specific GPU)
        torch.backends.cudnn.benchmark = True 
    else:
        device = torch.device("cpu")
        print("Warning: No GPU found. Running on CPU")
    dataset = SequenceDataset("./data/processed_data")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Initialize from scratch (No pretrained_cnn needed anymore)
    model = TemporalDetector(pretrained_cnn=None).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training Temporal Expert...")
    for epoch in range(1, 26):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        if total_loss/len(loader) < 0.13:
            print(">> Early Stopping: Model is starting to memorize.")
            break
        
    torch.save(model.state_dict(), "./models/temporal_model.pth")
    print("Temporal Expert Saved.")

if __name__ == "__main__":
    train_temporal_expert()