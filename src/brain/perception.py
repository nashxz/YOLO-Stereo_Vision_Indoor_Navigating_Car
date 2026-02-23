import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# COPY AND RUN THIS ON POWERSHELL (LAPTOP) TO START THE GSTREAMER RTSP SERVER:
# cmd.exe /c '"C:\Program Files\gstreamer\1.0\msvc_x86_64\bin\gst-launch-1.0.exe" -v udpsrc port=5000 ! application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96 ! rtph264depay ! decodebin ! videoconvert ! autovideosink sync=false'

try:
    cv2.destroyAllWindows()
except:
    pass

# --- CONFIGURATION ---
W, H = 424, 240  # Unified resolution for speed
FPS = 30
STOP_DISTANCE = 0.6  # 60cm

# Setup RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
profile = pipeline.start(config)

# --- NEW: Setup Depth-to-Color Alignment ---
align_to = rs.stream.color
align = rs.align(align_to)
# 


gst_out = (
    "appsrc ! videoconvert ! video/x-raw, format=I420 ! "
    "x264enc tune=zerolatency speed-preset=ultrafast threads=4 bitrate=1500 key-int-max=30 ! "
    "rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=192.168.1.227 port=5000 sync=false"
)

out = cv2.VideoWriter(gst_out, cv2.CAP_GSTREAMER, 0, FPS, (W, H), True)

# --- FIX 1: The GStreamer Safety Check ---
if not out.isOpened():
    print("WARNING: GStreamer pipeline failed to open. Is your RTSP server running on localhost:8554?")

# --- FIX 2: Load the compiled TensorRT Engine ---
model = YOLO('/home/group26/Active-Stereo-Vision-Deep-Learning-Fusion-for-Real-Time-Indoor-Navigation/src/brain/yolov8n.engine', task='detect')

# --- FIX 3: State flag for the warm-up print ---
first_inference = True 

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame: continue

        frame = np.asanyarray(color_frame.get_data())

        # 1. AI Layer (YOLO)
        if first_inference:
            print("-> Running first AI Inference (Allocating GPU memory, please wait)...")
            
        results = model(frame, verbose=False)
        
        if first_inference:
            print("-> Inference complete! Sprinting at full FPS...")
            first_inference = False
        
        # Draw YOLO boxes FIRST
        frame = results[0].plot() 
        
        # 2. Reflex Layer (Depth check in center of screen)
        depth_data = np.asanyarray(depth_frame.get_data())
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        
        center_y, center_x = H//2, W//2
        patch = depth_data[center_y-2:center_y+3, center_x-2:center_x+3] * depth_scale
        
        # Filter out 0.0 values (invalid data) and find the average distance
        valid_depths = patch[patch > 0]
        center_depth = np.mean(valid_depths) if len(valid_depths) > 0 else 0.0

        # Logic
        if 0 < center_depth < STOP_DISTANCE:
            cv2.putText(frame, "BRAKE: OBSTACLE DETECTED", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)

        # Visuals (Drawn on top of everything)
        cv2.rectangle(frame, (center_x-2, center_y-2), (center_x+2, center_y+2), (255, 0, 0), 1)
        cv2.putText(frame, f"Center Dist: {center_depth:.2f}m", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if out.isOpened():
            out.write(frame)

finally:
    pipeline.stop()
    if out.isOpened():
        out.release()