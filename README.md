# YOLOv11s Arrow Detection

This repository contains a custom-trained **YOLOv11s** object detection model for the International Rover Challenge's Autonomous Expedition (AutEx) mission. The model is designed to detect specific arrow signs to aid robotic navigation.

---

## ✨ Features

- **Real-time Performance**: Optimized for fast inference on GPU-enabled hardware.
- **Custom-Trained**: Fine-tuned on a dedicated dataset of arrow signs.
- **Robustness**: Designed to perform well under varying lighting conditions and camera angles.

---

## 🎯 Model Details

- **Model Architecture**: YOLOv11s (small)
- **Task**: Object Detection
- **Trained Classes**: `arrow_left`, `arrow_right`
- **Training Epochs**: 60
- **Input Image Size**: 640x640 pixels
- **Training Hardware**: NVIDIA RTX 3050 GPU

---

## 🚀 Getting Started

### Prerequisites

Set up a Python environment with the following dependencies. The recommended approach is to use a **Conda environment** to avoid conflicts.

```bash
# 1. Create a new Conda environment with a compatible Python version
conda create -n yolov11_gpu python=3.11 -y

# 2. Activate the new environment
conda activate yolov11_gpu

# 3. Install PyTorch with CUDA support (for GPU acceleration)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install Ultralytics and OpenCV
pip install ultralytics opencv-python
```

---

### Model Weights

The pre-trained model weights are provided as `best.pt`. This is the file you will use for all your detection tasks.

---

## ⚙️ Usage

The provided Python script (`detect_webcam.py`) demonstrates how to use your trained `best.pt` model for real-time object detection with a webcam.

```python
import cv2
from ultralytics import YOLO
import torch

# Load the trained model. Replace with the path to your 'best.pt' file.
model = YOLO('path/to/your/best.pt')

# Use GPU if available, otherwise use CPU
device_info = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device_info}")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = model.predict(source=frame, show=False, conf=0.5, device=device_info, stream=True)
    
    # Display results on the frame
    for r in results:
        annotated_frame = r.plot()
        cv2.imshow('YOLOv11 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Instructions

1. **Save the script**: Save the code above as `detect_webcam.py` in your project folder.
2. **Update the path**: In the script, change `path/to/your/best.pt` to the actual path of your model file.
3. **Run the script**: With your Conda environment active, run `python detect_webcam.py` from your terminal.

---

## 💡 Mission Context

This model was trained to identify specific arrow signs used in the Autonomous Mission of the International Rover Challenge. The rover is required to navigate through arrow signs and move toward a cone or checkpoint, making accurate arrow detection crucial for mission success.

---
