# 😷 Face Mask Detection

Real-time and single-image face mask detection using **MobileNetV2** transfer learning + **OpenCV DNN** Caffe SSD face detector.

---

## 📁 Project Structure

```
face-mask-detection/
│
├── dataset/
│   ├── with_mask/          ← training images WITH mask
│   └── without_mask/       ← training images WITHOUT mask
│
├── face_detector/
│   ├── deploy.prototxt             ← Caffe SSD config
│   └── res10_300x300_ssd_iter_140000.caffemodel   ← face detector weights
│
├── models/
│   ├── mask_detector.keras         ← trained classifier (generated)
│   └── training_plot.png           ← accuracy/loss plot (generated)
│
├── screenshots/                    ← auto-saved violation frames
│
├── train_mask_detector.py          ← Step 1: train the model
├── detect_mask_video.py            ← Step 2a: real-time webcam/video detection
├── detect_image.py                 ← Step 2b: single image detection
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone & create virtual environment
```bash
git clone https://github.com/your-repo/face-mask-detection.git
cd face-mask-detection

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download face detector files

Download the two Caffe SSD files and place them in `face_detector/`:

| File | Download |
|------|----------|
| `deploy.prototxt` | [GitHub — opencv/samples](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) |
| `res10_300x300_ssd_iter_140000.caffemodel` | [Direct link (10 MB)](https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel) |

```bash
mkdir face_detector
# Place both downloaded files inside face_detector/
```

---

## 🗂️ Dataset

Organise your face images as below before training:

```
dataset/
├── with_mask/       ← ~1000+ JPEG/PNG images of masked faces
└── without_mask/    ← ~1000+ JPEG/PNG images of unmasked faces
```

Recommended public datasets:
- [Face Mask Detection Dataset – Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- [MaskedFace-Net](https://github.com/cabani/MaskedFace-Net)

---

## 🏋️ Step 1 — Train the Model

```bash
python train_mask_detector.py
```

What it does:
- Loads images from `dataset/`
- Fine-tunes **MobileNetV2** (ImageNet weights, head frozen)
- Saves best model → `models/mask_detector.keras`
- Saves accuracy/loss plot → `models/training_plot.png`

Key config inside `train_mask_detector.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | `(224, 224)` | Input size for MobileNetV2 |
| `BATCH_SIZE` | `32` | Training batch size |
| `EPOCHS` | `20` | Max training epochs (early stopping applies) |
| `LEARNING_RATE` | `1e-4` | Adam optimizer LR |

---

## 🎥 Step 2a — Real-Time Detection (Webcam / Video)

```bash
# Default: webcam
python detect_mask_video.py

# Use a video file
python detect_mask_video.py --source path/to/video.mp4

# Disable auto screenshot on violation
python detect_mask_video.py --no-save
```

### On-screen controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Save screenshot manually |

### HUD overlay

- **Green box** → mask detected
- **Red box** → no mask detected
- Top panel shows live face count, mask count, violation count and FPS
- Screenshots of violations are auto-saved to `screenshots/` after **3 consecutive** no-mask frames

---

## 🖼️ Step 2b — Single Image Detection

```bash
# Show result window
python detect_image.py --image photo.jpg

# Show and always save annotated result
python detect_image.py --image photo.jpg --save

# Save only when a violation is found
python detect_image.py --image photo.jpg --screenshot
```

### Interactive window controls

| Key | Action |
|-----|--------|
| `S` | Save screenshot |
| Any other key | Close window |

### Terminal output example
```
  Face 1: Mask        with_mask=0.97  without_mask=0.03
  Face 2: No Mask     with_mask=0.12  without_mask=0.88

  ── Summary ──────────────────
  Total Faces : 2
  With Mask   : 1  ✅
  No Mask     : 1  ❌

  ⚠️  VIOLATION: 1 person(s) not wearing mask!
```

---

## ⚙️ Configuration Reference

All thresholds are defined at the top of each script:

| Variable | Default | Description |
|----------|---------|-------------|
| `CONF_THRESHOLD` | `0.50` | Minimum confidence to accept a detected face |
| `MASK_THRESHOLD` | `0.60 / 0.65` | Minimum confidence to accept mask classification |
| `IMG_SIZE` | `(224, 224)` | Input size fed to mask classifier |
| `MODEL_PATH` | `models/mask_detector.keras` | Path to trained model |

---

## 🏗️ Model Architecture

```
MobileNetV2 (ImageNet, frozen)
        ↓
AveragePooling2D (7×7)
        ↓
Flatten
        ↓
Dense(128, relu)
        ↓
Dropout(0.5)
        ↓
Dense(2, softmax)   →   [with_mask, without_mask]
```

Trained with: `binary_crossentropy` loss · `Adam` optimizer · `EarlyStopping` + `ReduceLROnPlateau`

---

## 🖥️ GPU Support

Replace `tensorflow` in `requirements.txt` with:

```bash
# TensorFlow 2.13+ (recommended)
pip install tensorflow[and-cuda]

# Or for older TF versions
pip install tensorflow-gpu==2.10.0
```

Ensure CUDA and cuDNN versions match your TensorFlow version:
[TF–CUDA compatibility table](https://www.tensorflow.org/install/source#gpu)

---

## 📋 Requirements Summary

| Package | Purpose |
|---------|---------|
| `tensorflow` | MobileNetV2 model + Keras training API |
| `opencv-python` | Video capture, DNN face detector, annotations |
| `numpy` | Array processing and image pre-processing |
| `scikit-learn` | Data splitting and classification metrics |
| `matplotlib` | Training curve visualisation |

---

## 🛠️ Troubleshooting

**`[ERROR] Face detector files missing`**
→ Download both Caffe files into `face_detector/` (see Installation step 3).

**`[ERROR] Mask model not found`**
→ Run `train_mask_detector.py` first to generate `models/mask_detector.keras`.

**`[WARN] Folder not found: dataset/with_mask`**
→ Create the dataset directory structure and add images before training.

**Low FPS on CPU**
→ Reduce frame resolution with `cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)` or use GPU.

**Poor accuracy**
→ Add more balanced training data; try lowering `LEARNING_RATE` to `5e-5`.

---

## 📄 License

MIT License — free to use, modify and distribute.
