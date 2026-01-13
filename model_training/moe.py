import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

# Import your Architecture
from model_architecture import MoE_Investigator

class DeepfakeConsolidatedDataset(Dataset):
    def __init__(self, root_dirs):
        """
        root_dirs: List of folders, e.g. ["data/processed_data/real", "data/processed_data/fake"]
        """
        self.files = []
        for d in root_dirs:
            self.files.extend(list(Path(d).rglob("*.pt")))
        
        # Shuffle to mix Real and Fake
        random.shuffle(self.files)
        print(f">> Found {len(self.files)} consolidated training samples.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # Load the dictionary saved by process_data.py
        data = torch.load(self.files[idx], weights_only=False)
        
        # 1. RGB Sequence: (32, 3, 256, 256)
        rgb_seq = data['rgb'] 
        
        # 2. RGB Middle Frame: (3, 256, 256)
        mid_idx = rgb_seq.shape[0] // 2
        rgb_mid = rgb_seq[mid_idx]
        
        # 3. PRNU: (1, 1, 256, 256) -> Squeeze to (1, 256, 256) if needed
        prnu = data['prnu']
        if prnu.dim() == 4: prnu = prnu.squeeze(0)
            
        # 4. Audio: (1, 1, 128, 128) -> Squeeze to (1, 128, 128)
        audio = data['audio']
        if audio.dim() == 4: audio = audio.squeeze(0)
            
        # 5. Label
        label = torch.tensor([data['label']], dtype=torch.float32)
        
        return rgb_mid, rgb_seq, prnu, audio, label


def train_the_investigator():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training on {device} ---")
    
    # 1. Load Data
    # Point this to where process_data.py saved your files
    train_dirs = ["data/processed_data/real", "data/processed_data/fake"]
    dataset = DeepfakeConsolidatedDataset(train_dirs)
    
    # Batch size can be small because the model is huge (Loaded with experts)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 2. Initialize the MoE System
    print("Initializing System & Loading Experts...")
    model = MoE_Investigator(
        temp_path="temporal_model.pth", 
        art_path="unet_artifact_hunter.pth", 
        noise_path="poc_model_256.pth",
        audio_path="audio_expert.pth" 
    ).to(device)
    
    # 3. Define Optimizer
    # CRITICAL: We only want to train the ROUTER. The experts should be frozen.
    # The MoE_Investigator class already freezes experts in __init__, but we filter here to be safe.
    optimizer = optim.Adam(model.router.parameters(), lr=0.001)
    
    # Binary Cross Entropy Loss (Real vs Fake)
    criterion = nn.BCELoss()
    
    # 4. Train
    epochs = 10
    print("Starting Router Training...")
    
    for epoch in range(1, epochs+1):
        total_loss = 0
        
        model.train() # Set Router to train mode
        
        for batch_idx, (mid, seq, prnu, audio, label) in enumerate(loader):
            # Move to GPU
            mid = mid.to(device)     # (B, 3, H, W)
            seq = seq.to(device)     # (B, T, 3, H, W)
            prnu = prnu.to(device)   # (B, 1, H, W)
            audio = audio.to(device) # (B, 1, H, W)
            label = label.to(device) # (B, 1)
            
            optimizer.zero_grad()
            
            # Forward Pass (The Router decides weights -> Experts run -> Verdict returned)
            verdict, weights = model(mid, seq, prnu, audio)
            
            # Calculate Loss
            loss = criterion(verdict, label)
            
            # Backprop (Updates ONLY the Router's decision making)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Optional: Print weights every 10 batches to watch it learn
            if batch_idx % 10 == 0:
                w = weights[0].detach().cpu().numpy()
                print(f"   [Batch {batch_idx}] Loss: {loss.item():.4f} | Weights: Temp={w[0]:.2f} Art={w[1]:.2f} Noise={w[2]:.2f} Audio={w[3]:.2f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | Average Loss: {avg_loss:.4f}")

    # 5. Save the Brain
    torch.save(model.router.state_dict(), "router_weights.pth")
    print("\nTraining Complete.")
    print(">> 'router_weights.pth' saved.")
    print(">> You can now run 'run.py' to use the full trained system.")

if __name__ == "__main__":
    train_the_investigator()