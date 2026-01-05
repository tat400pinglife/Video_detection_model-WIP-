import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import cv2
import random

# IMPORT THE MODEL
from model_architecture import ArtifactSegmentor

class RealFramesDataset(Dataset):
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.real_folder = self.root / "real"
        self.files = list(self.real_folder.rglob("*.pt"))
        
        if not self.files:
            print(f"CRITICAL ERROR: No .pt files found in {self.real_folder}")
        else:
            print(f"Found {len(self.files)} real samples for U-Net training.")
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        try:
            data = torch.load(path, weights_only=False)
            frame_idx = random.randint(0, 31)
            rgb = data['rgb'][frame_idx].permute(1, 2, 0).numpy()
            return rgb
        except:
            return np.zeros((256, 256, 3), dtype=np.float32)

def create_training_batch(real_batch):
    batch_size, h, w, c = real_batch.shape
    frames_A = real_batch.numpy()
    frames_B = np.roll(frames_A, shift=1, axis=0)
    
    blended_batch = []
    masks_batch = []
    
    for i in range(batch_size):
        imgA, imgB = frames_A[i], frames_B[i]
        mask = np.zeros((h, w), dtype=np.float32)
        
        num_points = random.randint(3, 6)
        points = []
        for _ in range(num_points):
            points.append([random.randint(0, w), random.randint(0, h)])
        cv2.fillPoly(mask, np.array([points], dtype=np.int32), (1.0))
        
        k_size = random.choice([15, 21, 31])
        mask = cv2.GaussianBlur(mask, (k_size, k_size), 10)
        mask = mask[:, :, np.newaxis]
        
        blended = mask * imgB + (1 - mask) * imgA
        noise = np.random.normal(0, 0.02, blended.shape).astype(np.float32)
        blended = blended + (mask * noise)
        
        blended_batch.append(blended)
        masks_batch.append(mask)
        
    inputs = torch.tensor(np.array(blended_batch), dtype=torch.float32).permute(0, 3, 1, 2)
    targets = torch.tensor(np.array(masks_batch), dtype=torch.float32).permute(0, 3, 1, 2)
    return inputs, targets

def train_unet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = RealFramesDataset("./data/frames")
    if len(dataset) == 0: return

    loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    
    # Initialize from imported class
    model = ArtifactSegmentor().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting U-Net Training...")
    
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        for real_batch in loader:
            inputs, masks = create_training_batch(real_batch)
            inputs, masks = inputs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), "unet_artifact_hunter.pth")
    print("U-Net Saved.")

if __name__ == "__main__":
    train_unet()