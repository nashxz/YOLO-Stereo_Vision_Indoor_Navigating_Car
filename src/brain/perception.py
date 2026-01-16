import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# --- CONFIGURATION ---
# Camera Resolution (Lower = Faster processing on Jetson)
WIDTH, HEIGHT = 640, 480
FPS = 30

# The "Reflex" Safety Zone (The area in front of the robot to scan)
# We ignore the edges and focus on the center where collisions happen.
# Values are in pixels.
ROI_X_MIN, ROI_X_MAX = 200, 440  # Middle 240 pixels width
ROI_Y_MIN, ROI_Y_MAX = 100, 380  # Middle 280 pixels height

# Distance Thresholds (Meters)
STOP_DISTANCE = 0.6  # If anything is closer than 60cm, STOP.

# --- 1. SETUP HARDWARE ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

# Start Streaming
profile = pipeline.start(config)

# [CRITICAL] Enable IR Emitter for Textureless Walls
# This ensures we get depth data even on white walls.
depth_sensor = profile.get_device().first_depth_sensor()
if depth_sensor.supports(rs.option.emitter_enabled):
    depth_sensor.set_option(rs.option.emitter_enabled, 1.0) # 1 = ON

# Align Object (Matches Depth to Color pixels)
align = rs.align(rs.stream.color)

# --- 2. SETUP AI ---
print("Loading YOLO Model...")
model = YOLO('yolov8n.pt') # Uses the lightest model

try:
    while True:
        # Get Frames
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame: continue

        # Convert to Numpy Arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO Inference
        results = model(color_image, verbose=False)
        detections = results[0].boxes

        # --- HYBRID LOGIC START ---

        if len(detections) > 0:
            # === SCENARIO A: YOLO SEES SOMETHING ===
            # The robot is "smart." It knows what the object is.
            mode_text = "MODE: AI (YOLO)"
            status_color = (0, 255, 0) # Green

            for box in detections:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                
                # Simple box drawing
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            # === SCENARIO B: YOLO SEES NOTHING (REFLEX LAYER) ===
            # The robot is "blind" to objects, but "aware" of physics.
            # We explicitly check the depth map for ANY obstacle.
            
            mode_text = "MODE: REFLEX (DEPTH ONLY)"
            
            # 1. Cut out the Safety Zone from the depth image
            safety_zone_depth = depth_image[ROI_Y_MIN:ROI_Y_MAX, ROI_X_MIN:ROI_X_MAX]
            
            # 2. Filter out 0s (Noise) to avoid false positives
            valid_pixels = safety_zone_depth[safety_zone_depth > 0]

            if valid_pixels.size > 0:
                # 3. Find the CLOSEST point in that zone
                # We use percentile (e.g., 1%) instead of min() to ignore random noise specs
                closest_dist_mm = np.percentile(valid_pixels, 1)
                closest_dist_m = closest_dist_mm / 1000.0

                # 4. Safety Logic
                if closest_dist_m < STOP_DISTANCE:
                    status_text = f"OBSTACLE: {closest_dist_m:.2f}m - STOP!"
                    status_color = (0, 0, 255) # Red
                    # TODO: Send 'STOP' to Arduino/ESP32 here
                else:
                    status_text = f"Path Clear ({closest_dist_m:.2f}m)"
                    status_color = (255, 255, 0) # Cyan
            else:
                # If all pixels are 0, we are either blind or looking at infinity
                status_text = "No Depth Data"
                status_color = (128, 128, 128)

            # Visualize the Safety Zone Box so you can see where it's looking
            cv2.rectangle(color_image, (ROI_X_MIN, ROI_Y_MIN), 
                          (ROI_X_MAX, ROI_Y_MAX), status_color, 2)
            cv2.putText(color_image, status_text, (ROI_X_MIN, ROI_Y_MIN - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # Draw Mode Label
        cv2.putText(color_image, mode_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display
        cv2.imshow('Capstone Hybrid Logic', color_image)
        if cv2.waitKey(1) == ord('q'): break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()