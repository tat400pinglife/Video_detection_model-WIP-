import subprocess
import sys
import os
import time

# flahs for each model
TRAIN_TEMPORAL  = False   # Motion / Ghosting
# TRAIN_ARTIFACT  = False   # Visual Glitches (RGB)
# TRAIN_NOISE     = False   # Camera Sensor Noise (PRNU)
# TRAIN_FREQ      = False   # Frequency Analysis (FFT)
# TRAIN_AUDIO     = False   # Audio Spectrum

#TRAIN_TEMPORAL  = True   # Motion / Ghosting
TRAIN_ARTIFACT  = True   # Visual Glitches (RGB)
TRAIN_NOISE     = True   # Camera Sensor Noise (PRNU)
TRAIN_FREQ      = True   # Frequency Analysis (FFT)
TRAIN_AUDIO     = True   # Audio Spectrum
# this will always be true
TRAIN_MOE_ROUTER = True


# Script mapping
SCRIPTS = {
    #"temporal": "temporal.py",
    "temporal": "model_training/temporalVlstm.py",
    "artifact": "model_training/AreaDetectionModel.py", 
    "noise":    "model_training/prnu.py",
    "freq":     "model_training/frequency.py",
    "audio":    "model_training/audio.py"   
}

MOE_SCRIPT = "model_training/moe2.py"         

def run_step(script_name, step_name):

    if not os.path.exists(script_name):
        print(f"!! [SKIP] {step_name}: Script '{script_name}' not found.")
        return False

    print(f"   STARTING: {step_name.upper()} TRAINING")
    print(f"   Script: {script_name}")
    print("="*60 + "\n")

    start_time = time.time()
    
    # Run the script as a subprocess
    try:
        # sys.executable ensures we use the same python environment (conda/venv)
        subprocess.run([sys.executable, script_name], check=True)
        
        duration = time.time() - start_time
        print(f"\n>> [SUCCESS] {step_name} finished in {duration:.1f}s.")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {step_name} failed with exit code {e.returncode}.")
        print("Check the error messages above.")
        return False

def main():
    print("Training Pipeline Starting...\n")
    
    # 1. Build the Queue based on Config
    queue = []
    
    if TRAIN_TEMPORAL: queue.append(("temporal", SCRIPTS["temporal"]))
    if TRAIN_ARTIFACT: queue.append(("artifact", SCRIPTS["artifact"]))
    if TRAIN_NOISE:    queue.append(("noise",    SCRIPTS["noise"]))
    if TRAIN_FREQ:     queue.append(("freq",     SCRIPTS["freq"]))
    if TRAIN_AUDIO:    queue.append(("audio",    SCRIPTS["audio"]))

    if not queue and TRAIN_MOE_ROUTER:
        print("No experts selected. Jumping straight to MoE Router training...")
    elif not queue and not TRAIN_MOE_ROUTER:
        print("Nothing selected to train. Exiting.")
        return

    # 2. Run Expert Training
    fail_count = 0
    
    for name, script in queue:
        success = run_step(script, name)
        if not success:
            fail_count += 1

    # 3. Run MoE Router (The Manager)
    if TRAIN_MOE_ROUTER:
        print("\n" + "#"*60)
        print("FINAL STEP: TRAINING MOE ROUTER")
        print("#"*60)
        
        if fail_count > 0:
            print(f"WARNING: {fail_count} experts failed to train.")
            print("The Router will use the old/existing weights for those experts.")
            time.sleep(2) 
            
        run_step(MOE_SCRIPT, "MoE Router")
    else:
        print("\n>> Skipping MoE Training (Disabled in Config).")

    print(f"\n Done")

if __name__ == "__main__":
    main()