import cv2
import os
import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

print("Root dir:", root_dir )
cap = cv2.VideoCapture(f"{root_dir}/data/videos/real/real_41003.mp4")
print("Successfully Opened:", cap.isOpened())
ret, frame = cap.read()
print("Successfully Read First Frame:", ret)