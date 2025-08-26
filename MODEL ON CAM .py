import cv2
import sys
from ultralytics import YOLO
import torch

# This script performs real-time object detection using your trained YOLO model
# with a webcam. It's the final step to see your model in action!

# 1. Load the trained model.
# IMPORTANT: Replace 'path/to/your/best.pt' with the actual path to your trained model file.
# Your model is likely located in a path similar to:
# 'C:/Users/SUJAL/Documents/MAIN/ARROW DETECTION MODEL/runs/detect/train/weights/best.pt'
model = YOLO("model\model 26-08-2025 11-10PM .pt")

# 2. Check for CUDA (GPU) availability and set the device.
# This will automatically use your GPU, which you have now correctly set up.
device_info = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_info}")

# 3. Set up video capture from the webcam.
# The argument '0' typically refers to the default webcam.
# You can change this to a video file path if you want to run detection on a video.
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully.
if not cap.isOpened():
    print("Error: Could not open video stream or file.")
    sys.exit()

# 4. Process video frames in a loop.
# This loop continuously captures frames, runs detection, and displays the results.
while True:
    # Read a new frame from the video capture.
    ret, frame = cap.read()
    if not ret:
        break # Break the loop if reading a frame fails.

    # 5. Perform object detection on the frame.
    # The 'stream=True' argument makes the prediction faster by processing the frame
    # as a stream. 'conf=0.5' is the confidence threshold. You can adjust this value
    # to control how confident the model must be to show a detection.
    results = model.predict(source=frame, show=False, conf=0.5, device=device_info, stream=True)
    
    # 6. Draw bounding boxes and display the annotated frame.
    # The 'plot()' method of the results object draws the detections for us.
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow('YOLOv11 Detection', annotated_frame)

    # 7. Exit the loop.
    # The loop will break if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 8. Release resources.
# Release the video capture object and close all OpenCV windows.
cap.release()
cv2.destroyAllWindows()
