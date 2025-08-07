# Object Detection for Home Security

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Key Components](#key-components)
- [How the Code Works](#how-the-code-works)
- [Performance Results](#performance-results)
- [Installation & Setup](#installation--setup)
- [Usage Instructions](#usage-instructions)
- [System Workflow](#system-workflow)
- [Key Features](#key-features)
- [Model Comparison](#model-comparison)
- [File Structure](#file-structure)
- [Future Improvements](#future-improvements)

## Project Overview

This project develops a **real-time home security system** that detects and classifies objects using live camera feeds. The system uses advanced YOLO (You Only Look Once) models to distinguish between normal and suspicious activity, specifically focusing on detecting dangerous objects like persons, knives, guns, and fires to trigger automated email alerts.

### Objectives
- Develop reliable automated surveillance for homeowners, renters, and small businesses
- Achieve high accuracy with minimal false alerts
- Provide real-time object detection with immediate notification system
- Compare multiple YOLO model versions for optimal performance

### Input/Output Summary
**Inputs:**
- Real-time webcam video or prerecorded footage
- COCO dataset (80 object classes) for training/testing

**Intermediate Outputs:**
- Detected bounding boxes and confidence scores
- Real-time annotated video stream

**Final Outputs:**
- Labeled objects in video frames
- Alert notifications (via email) upon detecting specific "danger" classes

## System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Live Camera   │───▶│   YOLO Model     │───▶│  Object         │
│   Feed (OpenCV) │    │   (YOLOv5)       │    │  Classification │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Email Alert   │◀───│  Danger Object   │◀───│  Threat         │
│   System (SMTP) │    │  Detection       │    │  Assessment     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Key Components

### 1. Models Implemented
- **YOLOv3** - Baseline implementation
- **YOLOv5** - Primary model (chosen for deployment)
- **YOLOv8** - Latest version comparison
- **COCO Dataset** - 80 object classes for training/testing

### 2. Core Modules
- **Real-time Detection** - Webcam integration using OpenCV
- **Model Training** - Comparative training across YOLO versions
- **Alert System** - Email notifications for dangerous objects
- **Performance Evaluation** - Comprehensive metrics analysis

## How the Code Works

### Step 1: Environment Setup
```python
!pip install -q ultralytics
from ultralytics import YOLO

# Dataset path configuration
DATA_PATH = "../input/coco-mini-5000"
TRAIN_IMAGES = f"{DATA_PATH}/images/train2017"
VAL_IMAGES = f"{DATA_PATH}/images/val2017"
TRAIN_LABELS = f"{DATA_PATH}/labels/train2017"
VAL_LABELS = f"{DATA_PATH}/labels/val2017"
```

### Step 2: Dataset Configuration
```python
yaml_content = """
path: /kaggle/input/coco-mini-4000/coco-mini-4000
train: images/train2017
val: images/val2017
nc: 80
names: [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
         'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
         'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
         'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
         'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
         'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
         'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
         'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
         'toothbrush' ]
"""

with open("coco-mini-4000.yaml", "w") as f:
    f.write(yaml_content)
```

### Step 3: Model Training & Comparison

#### YOLOv8 Training:
```python
model = YOLO("yolov8s.pt")
results = model.train(
    data="coco-mini-4000.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    project="kaggle_yolo_training",
    name="coco_mini_4000_run",
    save=True
)
```

#### YOLOv5 Training:
```python
!python train.py \
  --img 640 \
  --batch 8 \
  --epochs 15 \
  --data coco-mini-4000.yaml \
  --weights yolov5m.pt \
  --project yolov5_train \
  --name coco_mini_4000_run
```

#### YOLOv3 Training:
```python
!python train.py \
  --img 640 \
  --batch 8 \
  --epochs 15 \
  --data coco-mini-4000.yaml \
  --cfg models/yolov3.yaml \
  --weights '' \
  --name coco_mini_yolov3 \
  --project yolov3_train
```

### Step 4: Model Evaluation
```python
import os
import glob
from sklearn.metrics import classification_report

# Load trained model
model = YOLO("/path/to/best.pt")

# Dataset paths
val_img_dir = "/path/to/val/images"
val_lbl_dir = "/path/to/val/labels"

y_true = []
y_pred = []

# Process validation images
image_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))

for img_path in image_paths:
    # Load ground truth labels
    label_path = os.path.join(val_lbl_dir, os.path.basename(img_path).replace(".jpg", ".txt"))
    
    gt_classes = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls_id = int(float(line.split()[0]))
                gt_classes.append(cls_id)
    
    if not gt_classes:
        continue
    
    # Run inference
    result = model.predict(img_path, conf=0.25, iou=0.5, verbose=False)[0]
    pred_classes = result.boxes.cls.cpu().numpy().astype(int) if result.boxes else []
    
    # Match predictions with ground truth
    if len(pred_classes) >= len(gt_classes):
        y_true.extend(gt_classes)
        y_pred.extend(pred_classes[:len(gt_classes)])
    else:
        y_true.extend(gt_classes[:len(pred_classes)])
        y_pred.extend(pred_classes)

# Generate classification report
print(classification_report(y_true, y_pred, zero_division=0))
```

### Step 5: Real-Time Detection System

#### Webcam Integration:
```python
import cv2
import torch
from ultralytics import YOLO

# Load trained model
model = YOLO("path/to/best.pt")
model.conf = 0.4  # Confidence threshold

# Initialize webcam
cap = cv2.VideoCapture(0)

# Dangerous object classes
danger_classes = ["person", "knife", "gun", "fire"]
already_alerted = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model.predict(frame, conf=0.4)
    
    # Process detections
    for detection in results:
        boxes = detection.boxes
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls)
                class_name = model.names[cls_id]
                confidence = box.conf.item()
                
                # Draw bounding box
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Check for dangerous objects
                if class_name in danger_classes and not already_alerted:
                    send_alert_email(f"ALERT: Detected {class_name} on webcam!")
                    already_alerted = True
    
    # Display frame
    cv2.imshow('Security Camera', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

#### Email Alert System:
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert_email(message):
    sender_email = "your_email@gmail.com"
    receiver_email = "receiver@gmail.com"
    app_password = "your_app_password"
    
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Security Alert"
    
    msg.attach(MIMEText(message, 'plain'))
    
    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
        print("Alert email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
```

## Performance Results

### Model Comparison Table
| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Status |
|-------|-----------|--------|---------|--------------|---------|
| YOLOv3| 0.406     | 0.083  | 0.0532  | 0.0251       | ❌ Poor |
| YOLOv5| 0.675     | 0.560  | 0.609   | 0.416        | ✅ **Selected** |
| YOLOv8| 0.6551    | 0.5472 | 0.5871  | 0.4282       | ⚡ Good but slower |

### Key Findings:
- **YOLOv5 chosen** as optimal model for deployment
- **Best balance** of accuracy, speed, and consistency
- **Real-time performance** suitable for webcam streaming
- **Stable training curves** with reliable convergence

## Installation & Setup

### Prerequisites
```bash
pip install ultralytics opencv-python scikit-learn matplotlib pandas torch torchvision
```

### Additional Requirements
```bash
pip install smtplib email
```

### Dataset Setup
1. Download COCO mini dataset
2. Organize in the following structure:
```
coco-mini-4000/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

## Usage Instructions

### 1. Training Phase
```bash
# Run the Jupyter notebook sequentially
jupyter notebook "notebookce0b99c74d (2).ipynb"
```

### 2. Configuration Setup
Create `configuration.py`:
```python
# Email Configuration
SENDER_EMAIL = "your_security_camera@gmail.com"
RECEIVER_EMAIL = "your_phone@gmail.com"
APP_PASSWORD = "your_gmail_app_password"

# Detection Configuration
CONFIDENCE_THRESHOLD = 0.4
DANGER_CLASSES = ["person", "knife", "gun", "fire"]

# Camera Configuration
CAMERA_INDEX = 0  # 0 for default webcam
```

### 3. Real-Time Detection
```bash
python webcam.py
```

### 4. Model Evaluation
```bash
# Evaluate trained models
python evaluate_models.py
```

## System Workflow

1. **Camera Initialization** 📹
   - OpenCV captures live webcam feed
   - Frame preprocessing and resizing to 640x640

2. **Object Detection** 🔍
   - YOLOv5 processes each frame
   - Confidence threshold filtering (0.4)
   - Bounding box generation with class labels

3. **Threat Assessment** ⚠️
   - Check detected objects against danger classes
   - Evaluate confidence scores
   - Trigger alert system if threats found

4. **Alert System** 📧
   - Email notification sent via Gmail SMTP
   - Anti-spam mechanism (one alert per session)
   - Alert message: "ALERT: Detected [object] on webcam!"

5. **Visual Feedback** 📺
   - Real-time video display with bounding boxes
   - Class labels and confidence scores
   - Color-coded detection indicators

## Key Features

### ✅ Core Capabilities
- **Real-time Processing** - Live webcam feed analysis
- **Multi-Model Support** - YOLOv3, YOLOv5, YOLOv8 comparison
- **Intelligent Alerting** - Selective notifications for dangerous objects
- **Email Integration** - SMTP-based notification system
- **Visual Feedback** - Bounding boxes with confidence scores
- **Anti-Spam Protection** - One alert per detection session

### 🛡️ Security Features
- **Dangerous Object Detection** - Person, knife, gun, fire recognition
- **Confidence Thresholding** - Reduces false positive alerts
- **Real-time Monitoring** - Continuous surveillance capability
- **Remote Notifications** - Email alerts for off-site monitoring

### 📊 Performance Features
- **Model Benchmarking** - Comprehensive performance comparison
- **Metrics Analysis** - Precision, recall, mAP evaluation
- **Training Visualization** - Loss curves and metric tracking
- **Classification Reports** - Per-class performance analysis

## Model Comparison

### YOLOv3 Analysis
- **Performance**: Poor (40.6% precision, 8.3% recall)
- **Issues**: Failed to detect small/occluded objects
- **Status**: Not suitable for deployment
- **Reason**: Older architecture lacks modern enhancements

### YOLOv5 Analysis ⭐ (Selected)
- **Performance**: Excellent (67.5% precision, 56% recall)
- **Advantages**: 
  - Best balance of speed and accuracy
  - Stable training curves
  - Reliable real-time performance
  - Consistent object localization
- **Status**: **Primary deployment model**

### YOLOv8 Analysis
- **Performance**: Very Good (65.5% precision, 54.7% recall)
- **Advantages**: Highest mAP@0.5:0.95 score
- **Disadvantages**: Slightly slower real-time performance
- **Status**: Good alternative but heavier architecture

## File Structure

```
Object-detection-for-home-security/
├── notebookce0b99c74d (2).ipynb    # Main implementation notebook
├── final report cvpi.pdf            # Comprehensive project report
├── PROGRESS REPORT 1.pdf            # Initial progress documentation
├── Progress_report-2.pdf            # Second progress update
├── Project proposal.pdf             # Original project proposal
├── README.md                        # This documentation file
├── Screenshots/                     # Output visualization images
│   ├── Screenshot 2025-05-08...png
│   └── tmp*.PNG
├── weights/                         # Trained model weights (generated)
│   ├── yolov5_best.pt
│   ├── yolov8_best.pt
│   └── yolov3_best.pt
└── configs/                         # Configuration files (to be created)
    ├── configuration.py
    ├── webcam.py
    └── coco-mini-4000.yaml
```

## Future Improvements

### 🔮 Planned Enhancements
1. **Model Quantization** - Edge deployment optimization
2. **Behavior Analysis** - Advanced threat assessment logic
3. **Multi-Camera Support** - Multiple feed monitoring
4. **Cloud Integration** - Remote model deployment
5. **Mobile App** - Smartphone notification system

### 🛠️ Technical Improvements
1. **Performance Optimization** - Faster inference speed
2. **Memory Efficiency** - Reduced resource consumption
3. **Alert Customization** - User-defined threat categories
4. **Historical Analysis** - Event logging and analysis
5. **Integration APIs** - Third-party system connectivity

### 📱 User Experience
1. **Web Dashboard** - Browser-based monitoring interface
2. **Alert Scheduling** - Time-based notification rules
3. **False Positive Reduction** - Advanced filtering algorithms
4. **Custom Training** - User-specific object recognition
5. **Privacy Features** - Local processing options

---

## 📞 Support & Documentation

For questions, issues, or contributions:
- Review the comprehensive project report: `final report cvpi.pdf`
- Check progress documentation: `PROGRESS REPORT 1.pdf` and `Progress_report-2.pdf`
- Examine the complete implementation: `notebookce0b99c74d (2).ipynb`

## 📄 License

This project is developed for educational and research purposes. Please ensure compliance with local privacy laws when deploying surveillance systems.

---

**Project Status**: ✅ Complete and Deployable  
**Last Updated**: 2025  
**Model Recommendation**: YOLOv5 for optimal performance balance