# create_tensors.py
from imports.utils import process_dataset

if __name__ == "__main__":
    REAL_VIDEO_DIR = "./data/videos/real"
    FAKE_VIDEO_DIR = "./data/videos/fake"
    
    print("Starting Tensor Creation")
    
    print(f"\nProcessing REAL videos from: {REAL_VIDEO_DIR}")
    process_dataset(REAL_VIDEO_DIR, "./data/processed_data/real", label=0.0)
    
    print(f"\nProcessing FAKE videos from: {FAKE_VIDEO_DIR}")
    process_dataset(FAKE_VIDEO_DIR, "./data/processed_data/fake", label=1.0)
    
    print("\n--- Processing Complete ---")