import torch
from model_training.model_architecture import MoE_Investigator

def count_parameters(model):
    """Returns the total number of trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def audit_system_parameters():
    print("Initializing Deepfake Investigation Unit Parameter Audit...\n")
    
    # Initialize the system (on CPU so it doesn't waste VRAM)
    system = MoE_Investigator().to(torch.device("cpu"))
    
    # Unfreeze everything temporarily just for the count
    # (Since MoE_Investigator freezes experts in its __init__)
    for param in system.parameters():
        param.requires_grad = True

    # Count parameters for each specific component
    temp_params = count_parameters(system.expert_temp)
    art_params = count_parameters(system.expert_art)
    freq_params = count_parameters(system.expert_freq)
    audio_params = count_parameters(system.expert_audio)
    
    # PRNU is split between the net and the head in your architecture
    prnu_params = count_parameters(system.expert_noise_net) + count_parameters(system.expert_noise_head)
    
    router_params = count_parameters(system.router)
    
    total_params = temp_params + art_params + freq_params + audio_params + prnu_params + router_params

    # Print the verification table
    print(f"{'Component':<25} | {'Parameter Count':>15}")
    print("-" * 43)
    print(f"{'1. Temporal (LSTM)':<25} | {temp_params:>15,}")
    print(f"{'2. Artifact Segmentor':<25} | {art_params:>15,}")
    print(f"{'3. Frequency (FFT)':<25} | {freq_params:>15,}")
    print(f"{'4. Audio Expert':<25} | {audio_params:>15,}")
    print(f"{'5. PRNU (Noise)':<25} | {prnu_params:>15,}")
    print(f"{'6. MoE Router':<25} | {router_params:>15,}")
    print("-" * 43)
    print(f"{'TOTAL SYSTEM PARAMETERS':<25} | {total_params:>15,}")
    
    # Calculate rough VRAM footprint (Assuming 32-bit floats = 4 bytes per param)
    mb_size = (total_params * 4) / (1024 * 1024)
    print(f"\n>> System VRAM Base Footprint: ~{mb_size:.2f} MB")

if __name__ == "__main__":
    audit_system_parameters()