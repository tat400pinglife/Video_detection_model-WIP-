import os
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path

# TODO: eventually will need to parallelize this since there is 500k videos, or 6mil dont shoot the messenger
#       also will need to convert to tensors so the model can actually process them
#       also will need to find out the normalized size of dataset, right now im just testing own videos
#       bouta eat smth maybe take a break

def extract_rgb_frames(video_path, out_dir, fps=5, size=224):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={size}:{size}",
        f"{out_dir}/frame_%05d.jpg"
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted(Path(out_dir).glob("frame_*.jpg"))
    return frames

def compute_temporal_diff(frames, diff_dir):
    os.makedirs(diff_dir, exist_ok=True)
    prev_frame = None
    for i, frame_path in enumerate(frames):
        frame = np.array(Image.open(frame_path).convert('L'), dtype=np.float32)
        if prev_frame is not None:
            diff = np.abs(frame - prev_frame).astype(np.uint8)
            Image.fromarray(diff).save(f"{diff_dir}/diff_{i:05d}.jpg")
        prev_frame = frame

def compute_spatial_fft(frames, fft_dir):
    os.makedirs(fft_dir, exist_ok=True)
    for i, frame_path in enumerate(frames):
        frame = np.array(Image.open(frame_path).convert('L'), dtype=np.float32)
        fft = np.fft.fft2(frame)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        magnitude = 255 * (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        Image.fromarray(magnitude.astype(np.uint8)).save(f"{fft_dir}/fft_{i:05d}.jpg")

def process_video(video_path, base_out_dir, fps=5, size=224):
    video_name = Path(video_path).stem
    rgb_dir = Path(base_out_dir) / video_name / "rgb"
    diff_dir = Path(base_out_dir) / video_name / "diff"
    fft_dir = Path(base_out_dir) / video_name / "fft"

    print(f"Processing video: {video_name}")
    frames = extract_rgb_frames(video_path, rgb_dir, fps=fps, size=size)
    compute_temporal_diff(frames, diff_dir)
    compute_spatial_fft(frames, fft_dir)
    print(f"Done: {video_name} -> {len(frames)} frames extracted")


videos_dir = Path("data/videos")
output_dir = Path("data/frames")
fps = 24
size = 400

for video_file in videos_dir.glob("*.mp4"):
    process_video(video_file, output_dir, fps=fps, size=size)
