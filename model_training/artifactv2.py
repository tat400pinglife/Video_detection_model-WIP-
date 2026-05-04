import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt

# Stop OpenCV from fighting with PyTorch Multiprocessing
cv2.setNumThreads(0)

# Import your architecture
from model_architecture import ArtifactSegmentor

# --- CONFIGURATION
DATA_PATH = "./data/processed_data" 
SAVE_DIR = "./models"   
BATCH_SIZE = 32                     
LR = 0.0001                           
EPOCHS = 15
NUM_WORKERS = 4                   
PIN_MEMORY = True                

# 1. GLITCH GENERATOR 
def create_glitch_batch_fast(real_imgs_numpy):
    """
    Optimized glitch generation. Runs inside DataLoader workers.
    Input: Numpy batch (B, H, W, 3)
    Returns: Tensor Inputs (B, 3, H, W), Tensor Masks (B, 1, H, W)
    """
    # Normalize if needed
    if real_imgs_numpy.max() > 1.0:
        real_imgs_numpy = real_imgs_numpy.astype(np.float32) / 255.0
        
    batch_size, h, w, c = real_imgs_numpy.shape
    inputs = np.zeros_like(real_imgs_numpy)
    masks = np.zeros((batch_size, h, w), dtype=np.float32)
    
    for i in range(batch_size):
        img = real_imgs_numpy[i]
        
        # Fast Randoms
        cx, cy = np.random.randint(50, 200, 2)
        
        # A. Create Mask (Vectorized where possible or simple CV2)
        mask = np.zeros((h, w), dtype=np.float32)
        if np.random.rand() > 0.5:
            radius = np.random.randint(30, 80)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        else:
            size = np.random.randint(40, 100)
            x1, y1 = max(0, cx - size//2), max(0, cy - size//2)
            cv2.rectangle(mask, (x1, y1), (x1+size, y1+size), 1.0, -1)
            
        # Blur mask to avoid sharp edges being the only cue
        mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)[:,:,None]
        
        # B. Create Artifact (Pixelation)
        scale = np.random.uniform(0.1, 0.4)
        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        artifact = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Color Shift
        c_idx = np.random.randint(0, 3)
        artifact[:, :, c_idx] *= np.random.uniform(0.7, 1.3)
        artifact = np.clip(artifact, 0, 1)

        # C. Blend
        blended = img * (1 - mask_blur) + artifact * mask_blur
        
        inputs[i] = blended
        masks[i]  = mask

    # Convert to Tensor
    t_inputs = torch.from_numpy(inputs).permute(0,3,1,2).float()
    t_masks  = torch.from_numpy(masks).unsqueeze(1).float()
    
    return t_inputs, t_masks

# 2. ROBUST DATASET
class BigDataTensorDataset(Dataset):
    def __init__(self, root_dir):
        self.files = list(Path(root_dir).rglob("*.pt"))
        if len(self.files) == 0:
            print(f"ERROR: No .pt files found in {root_dir}. Searching recursive...")
            self.files = list(Path(root_dir).rglob("real/*.pt"))
            
        print(f">> Indexing complete. Found {len(self.files)} files.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Retry logic for corrupt files
        attempts = 0
        while attempts < 3:
            try:
                path = self.files[idx]
                data = torch.load(path, weights_only=False)
                
                # Check for 'rgb_batch' (Sequence) or fallback to 'rgb_mid'
                if 'rgb_batch' in data:
                    frames = data['rgb_batch']
                    # Pick random frame index
                    ridx = np.random.randint(0, frames.shape[0])
                    frame = frames[ridx] # [3, 256, 256]
                elif 'rgb_mid' in data:
                    frame = data['rgb_mid']
                else:
                    raise ValueError("No RGB data in file")

                # Permute to HWC for OpenCV processing in the collate_fn or training loop
                # frame is [3, H, W] -> [H, W, 3]
                return frame.permute(1, 2, 0).numpy()

            except Exception as e:
                # If file is bad, pick a random OTHER file
                idx = np.random.randint(0, len(self.files))
                attempts += 1
        
        # If all fails, return zeros
        return np.zeros((256, 256, 3), dtype=np.float32)

# 3. CUSTOM COLLATE FUNCTION
def glitch_collate_fn(batch):
    # Filter out bad samples (zeros)
    clean_batch = [x for x in batch if x.max() > 0]
    if len(clean_batch) == 0:
        return None, None
        
    # Stack into numpy array (B, H, W, 3)
    batch_np = np.stack(clean_batch)
    
    # Generate Glitches (CPU Parallelized)
    inputs, targets = create_glitch_batch_fast(batch_np)
    
    return inputs, targets

# 4. MAIN TRAINING LOOP 
def train_scale():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs("./plots", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Large Scale Training on {device} ---")
    
    # Data Setup
    full_ds = BigDataTensorDataset(DATA_PATH)
    
    # Split 90/10
    train_len = int(0.9 * len(full_ds))
    val_len = len(full_ds) - train_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len])
    
    # Loaders with Multiprocessing
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        collate_fn=glitch_collate_fn,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY,
        collate_fn=glitch_collate_fn,
        drop_last=True
    )

    # Model Setup
    model = ArtifactSegmentor().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Reduce LR if validation loss stops improving for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    
    # Graph Tracking
    train_loss_data = []
    val_loss_data = []
    val_acc_data = []
    training_acc_data = []
    
    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch}/{EPOCHS} [Train]")
        
        for inputs, masks in loop:
            if inputs is None: continue # Skip bad batches
            
            inputs, masks = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Pixel-wise accuracy
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == masks).sum().item()
            train_total += masks.numel()
            
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = 100 * train_correct / train_total
        
        train_loss_data.append(avg_train_loss)
        training_acc_data.append(avg_train_acc)
        
        # VALIDATE
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, masks in val_loader:
                if inputs is None: continue
                inputs, masks = inputs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                
                logits = model(inputs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == masks).sum().item()
                val_total += masks.numel()
                
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = 100 * val_correct / val_total
        
        val_loss_data.append(avg_val_loss)
        val_acc_data.append(avg_val_acc)
        
        # Logging
        print(f"Results: Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        
        # Step Scheduler
        scheduler.step(avg_val_loss)
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save exactly where moe2.py expects to find it
            torch.save(model.state_dict(), f"{SAVE_DIR}/artifact_model.pth")
            print(">>> New Best Model Saved!")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_data, label='Training Loss')
    plt.plot(val_loss_data, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./plots/artifact_model_training_loss.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(training_acc_data, label='Training Accuracy')
    plt.plot(val_acc_data, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig("./plots/artifact_model_training_accuracy.png")
    plt.close()

    print("Done.")

if __name__ == "__main__":
    # Windows needs this for multiprocessing
    torch.multiprocessing.freeze_support()
    train_scale()