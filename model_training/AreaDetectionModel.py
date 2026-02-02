import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import random
from pathlib import Path

# Import your architecture
from model_architecture import ArtifactSegmentor

# --- 1. THE GLITCH GENERATOR (Unchanged, this logic is good) ---
def create_glitch_batch(real_imgs_batch):
    """
    Takes a batch of REAL images (B, H, W, 3) and creates FAKES + MASKS.
    """
    # Ensure inputs are 0.0 - 1.0
    if real_imgs_batch.max() > 1.0:
        real_imgs_batch = real_imgs_batch / 255.0
        
    batch_size, h, w, c = real_imgs_batch.shape
    inputs, masks = [], []
    
    for i in range(batch_size):
        img = real_imgs_batch[i] # (H, W, 3) Float
        
        # A. Create Mask
        mask = np.zeros((h, w), dtype=np.float32)
        cx, cy = random.randint(50, 200), random.randint(50, 200)
        
        if random.random() > 0.5:
            radius = random.randint(30, 80)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        else:
            size = random.randint(40, 100)
            x1, y1 = max(0, cx - size//2), max(0, cy - size//2)
            cv2.rectangle(mask, (x1, y1), (x1+size, y1+size), 1.0, -1)
            
        mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)[:,:,None]
        
        # B. Create Artifact
        scale = random.uniform(0.1, 0.3)
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        artifact = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        c_idx = random.randint(0, 2)
        artifact[:, :, c_idx] *= random.uniform(0.8, 1.2)
        artifact = np.clip(artifact, 0, 1)

        # C. Blend
        blended = img * (1 - mask_blur) + artifact * mask_blur
        
        # Add Noise
        noise = np.random.normal(0, 0.005, blended.shape).astype(np.float32)
        blended = np.clip(blended + noise, 0, 1)
        
        inputs.append(blended)
        masks.append(mask)

    t_inputs = torch.tensor(np.array(inputs)).permute(0,3,1,2).float()
    t_masks = torch.tensor(np.array(masks)).unsqueeze(1).float()
    
    return t_inputs, t_masks

# --- 2. FIXED DATASET CLASS ---
class RealTensorDataset(Dataset):
    def __init__(self, root_dir):
        # Look for files in processed_data, specifically REAL videos
        self.files = list(Path(root_dir).rglob("*.pt"))
        
        if len(self.files) == 0:
            print(f"ERROR: No .pt files found in {root_dir}")
            # Fallback check to prevent crash if user points to parent folder
            self.files = list(Path(root_dir).rglob("real/*.pt"))
            
        print(f">> Found {len(self.files)} real video tensors.")
        
    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            
            # FIX 1: Use the correct key 'rgb_batch'
            if 'rgb_batch' in data:
                frames_tensor = data['rgb_batch'] # Shape [32, 3, 256, 256]
            else:
                # Fallback if using older data processing
                frames_tensor = data['rgb_mid'].unsqueeze(0) 

            # FIX 2: Ensure we have frames to pick from
            num_frames = frames_tensor.shape[0]
            random_idx = random.randint(0, num_frames - 1)
            frame = frames_tensor[random_idx] # (3, 256, 256)
            
            # Permute for OpenCV: (3, H, W) -> (H, W, 3)
            frame = frame.permute(1, 2, 0)
            
            return frame 
            
        except Exception as e:
            # Return black frame on error
            return torch.zeros((256, 256, 3), dtype=torch.float32)

# --- 3. TRAINING LOOP ---
def train_artifact_expert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found device: {device}")
    
    # Point this to the folder containing REAL videos only
    # The glitch function creates the fakes for us
    dataset = RealTensorDataset("./data/processed_data") 
    
    # Drop last to prevent batch size errors
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    model = ArtifactSegmentor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Starting Self-Supervised Artifact Training...")
    
    epochs = 30
    for epoch in range(1, epochs+1):
        total_loss = 0
        model.train()
        
        count = 0
        for real_batch in loader:
            # Skip empty batches
            if real_batch.sum() == 0: continue

            # A. Create Fakes (Self-Supervision)
            inputs, masks = create_glitch_batch(real_batch.numpy())
            
            inputs = inputs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # B. Train
            logits = model(inputs) 
            loss = criterion(logits, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        if count == 0: continue
        avg_loss = total_loss / count
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f}")
        
        # Save periodically
        if epoch % 5 == 0:
             torch.save(model.state_dict(), "./models/artifact_model.pth")

    torch.save(model.state_dict(), "./models/artifact_model.pth")
    print("\n>> Training Complete. Saved 'artifact_model.pth'")

if __name__ == "__main__":
    train_artifact_expert()