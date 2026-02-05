## **Facial Recognition System**

A complete real-time facial recognition system based on Eigenfaces and OpenCV, featuring an interactive camera interface and statistical analysis tools.

---

## **Key Features**

* Real-time camera-based facial recognition
* Multiple face detection and identification
* Statistical analysis with charts and metrics
* Flexible configuration with adjustable parameters
* Optimized performance (25–30 FPS)
* Recognition accuracy between 80% and 95%

---

## **Table of Contents**

* Installation
* Usage
* Architecture
* Configuration
* Performance
* Troubleshooting
* Project Files

---

## **Installation**

### **Prerequisites**

* Python 3.8 or higher
* Webcam
* 500 MB of free disk space

### **Steps**

Clone or access the project directory:

Install dependencies:

```bash
pip install opencv-python numpy scikit-learn matplotlib seaborn
```

Verify installation:

```bash
python test_setup.py
```

---

## **Usage**

### **Option 1: Interactive Menu (Recommended)**

```bash
python main.py
```

Available options:

1. Full analysis (charts and statistics)
2. Real-time camera recognition
3. Exit

### **Option 2: Direct Camera Launch**

```bash
python run_camera.py
```

### **Option 3: Python Integration**

```python
from camera_capture import run_face_recognition_camera

run_face_recognition_camera(
    dataset_path="./face_database",
    n_components=50,
    threshold=5000
)
```

---

## **Camera Controls**

| Key    | Action                      |
| ------ | --------------------------- |
| q      | Quit and display statistics |
| Others | No effect                   |

---

## **On-Screen Display**

* Green rectangle: recognized face
* Red rectangle: unknown face
* Confidence distance displayed per face
* FPS, number of detected, recognized, and unknown faces shown in real time

---

## **Architecture**

```
Face_recognition/
│
├── main.py              # Main entry point
├── camera_capture.py    # Recognition engine
├── config.py            # Central configuration
├── run_camera.py        # Direct camera execution
├── test_setup.py        # Installation test
│
├── face_database/       # Face dataset
│   ├── s1/
│   ├── s2/
│   └── ...
│
├── README.md
├── README_CAMERA.md
├── UTILISATION.md
└── RESUME.md
```

---

## **Configuration**

### **Main Parameters**

Location: `camera_capture.py → run_face_recognition_camera()`

```python
run_face_recognition_camera(
    dataset_path="./face_database",
    n_components=50,
    threshold=5000
)
```

| Parameter    | Description                      |
| ------------ | -------------------------------- |
| dataset_path | Path to the face dataset         |
| n_components | Number of Eigenfaces (20–100)    |
| threshold    | Maximum distance for recognition |

### **Recommended Settings**

* **Higher accuracy (slower)**
  `n_components=100`, `threshold=3500`

* **Higher speed (less accurate)**
  `n_components=30`, `threshold=7000`

* **Balanced (recommended)**
  `n_components=50`, `threshold=5000`

---

## **Dataset Configuration**

Required structure:

```
face_database/
├── s1/
├── s2/
└── s3/
```

Recommendations:

* 8–12 images per person
* JPG or PNG format
* Resolution between 100×100 and 500×500
* Multiple angles and expressions
* Good lighting conditions

---

## **Performance**

### **Typical Results**

| Metric                | Value       |
| --------------------- | ----------- |
| Training time         | 2–3 seconds |
| Recognition per frame | 10–50 ms    |
| Real-time FPS         | 25–30       |
| Accuracy              | 80–95%      |
| Memory usage          | 200–400 MB  |

### **Optimization Tips**

* Reduce `n_components`
* Increase `threshold`
* Lower camera resolution
* Close background applications

---

## **Troubleshooting**

### **Camera Not Opening**

```bash
ls /dev/video*
sudo apt-get install cheese
cheese
sudo usermod -a -G video $USER
```

### **Missing Modules**

```bash
pip install scikit-learn seaborn
```

### **Low Accuracy**

* Increase `n_components`
* Decrease `threshold`
* Add more training images
* Improve lighting and image quality

### **Low FPS**

* Reduce `n_components`
* Close other applications
* Monitor CPU and RAM usage

---

## **How Facial Recognition Works**

1. Camera image capture (up to 30 FPS)
2. Face detection using Haar Cascade Classifier
3. Image preprocessing (resize and normalization)
4. Recognition using a pre-trained Eigenfaces model
5. Euclidean distance computation
6. Real-time visual feedback

### **Eigenfaces Algorithm**

**Advantages**

* Fast execution
* Low memory usage
* Suitable for small datasets

**Limitations**

* Sensitive to lighting conditions
* Less accurate than deep learning models

---

## **Future Improvements**

* Model saving and loading
* Deep learning integration (FaceNet, ArcFace)
* Graphical interface (PyQt or Tkinter)
* SQLite database
* Multi-threading
* Multi-camera support
* Video recording with annotations
* Statistics export

---
