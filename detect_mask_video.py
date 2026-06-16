"""
Face Mask Detection - Fixed Detection Script
- Prints raw prediction scores so you can see exactly what model outputs
- Correct class index mapping
- Higher confidence threshold to avoid misclassification
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os
import time
from datetime import datetime
import argparse

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE         = (224, 224)
CONF_THRESHOLD   = 0.50    # face detector minimum confidence
MASK_THRESHOLD   = 0.70    # mask classifier — must be 70%+ sure
MODEL_PATH       = "models/mask_detector.keras"
FACE_PROTO       = "face_detector/deploy.prototxt"
FACE_WEIGHTS     = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
SCREENSHOTS_DIR  = "screenshots"

COLOR_MASK       = (0, 200, 80)
COLOR_NO_MASK    = (0, 60, 220)
COLOR_HUD        = (255, 255, 255)

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ─── Load models ──────────────────────────────────────────────────────────────
def load_models():
    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_WEIGHTS):
        print("[ERROR] Face detector files missing in face_detector/ folder.")
        exit(1)
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)
    print("[INFO]  Face detector loaded.")

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Mask model not found at {MODEL_PATH}")
        print("        Run  py train_mask_detector.py  first.")
        exit(1)
    mask_net = load_model(MODEL_PATH)
    print("[INFO]  Mask classifier loaded.")

    # ── Print class mapping so we know which index = which class ─────────────
    # ImageDataGenerator.flow_from_directory sorts classes alphabetically:
    # with_mask = index 0,  without_mask = index 1
    print("[INFO]  Class mapping (alphabetical):")
    print("          index 0 → with_mask    (Mask ✅)")
    print("          index 1 → without_mask (No Mask ❌)\n")

    return face_net, mask_net

# ─── Detect faces ─────────────────────────────────────────────────────────────
def detect_faces(frame, face_net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONF_THRESHOLD:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w-1, x2); y2 = min(h-1, y2)
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue
        faces.append((x1, y1, x2, y2))
    return faces

# ─── Classify face ────────────────────────────────────────────────────────────
def classify_face(face_crop, mask_net, debug=False):
    """
    Returns (label, confidence).
    Class indices (alphabetical from ImageDataGenerator):
      0 = with_mask    → label "Mask"
      1 = without_mask → label "No Mask"
    """
    face = cv2.resize(face_crop, IMG_SIZE)
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    preds = mask_net.predict(face, verbose=0)[0]

    # preds[0] = probability of with_mask
    # preds[1] = probability of without_mask
    with_mask_prob    = float(preds[0])
    without_mask_prob = float(preds[1])

    if debug:
        print(f"  with_mask={with_mask_prob:.3f}  "
              f"without_mask={without_mask_prob:.3f}")

    # Only classify if model is confident enough
    max_prob = max(with_mask_prob, without_mask_prob)
    if max_prob < MASK_THRESHOLD:
        return None, max_prob   # not confident — skip

    if with_mask_prob > without_mask_prob:
        return "Mask", with_mask_prob
    else:
        return "No Mask", without_mask_prob

# ─── Draw box ─────────────────────────────────────────────────────────────────
def draw_annotation(frame, x1, y1, x2, y2, label, conf):
    color = COLOR_MASK if label == "Mask" else COLOR_NO_MASK
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label}: {conf*100:.0f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 0.6, 1)
    ly = max(y1 - th - 10, 0)
    cv2.rectangle(frame, (x1, ly), (x1 + tw + 10, ly + th + 10), color, -1)
    cv2.putText(frame, text, (x1 + 5, ly + th + 5),
                font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# ─── HUD ──────────────────────────────────────────────────────────────────────
def draw_hud(frame, total, with_mask, without_mask, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "FACE MASK DETECTION",
                (10, 25), font, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces : {total}",
                (10, 50), font, 0.65, COLOR_HUD, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Mask    : {with_mask}",
                (10, 72), font, 0.65, COLOR_MASK, 2, cv2.LINE_AA)
    cv2.putText(frame, f"No Mask : {without_mask}",
                (w // 2, 72), font, 0.65, COLOR_NO_MASK, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {fps:.0f}",
                (w - 80, h - 15), font, 0.55, (140, 140, 140), 1, cv2.LINE_AA)

# ─── Screenshot ───────────────────────────────────────────────────────────────
def save_screenshot(frame, count):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOTS_DIR, f"violation_{count}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVED] {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def run(source, save_violations=True, debug=False):
    face_net, mask_net = load_models()

    cap = cv2.VideoCapture(0 if source == "webcam" else source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}")
        exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO]  Source: {source}  {fw}x{fh}")
    print("        Q=quit  S=screenshot  D=toggle debug\n")

    total_violations = 0
    consec = {}
    fps_t  = time.time()
    fc     = 0
    fps    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO]  Stream ended.")
            break

        faces = detect_faces(frame, face_net)

        with_mask_count    = 0
        without_mask_count = 0

        for slot, (x1, y1, x2, y2) in enumerate(faces):
            face_crop   = frame[y1:y2, x1:x2]
            label, conf = classify_face(face_crop, mask_net, debug=debug)

            if label is None:
                consec[slot] = 0
                continue

            draw_annotation(frame, x1, y1, x2, y2, label, conf)

            if label == "Mask":
                with_mask_count += 1
                consec[slot] = 0
            else:
                without_mask_count += 1
                consec[slot] = consec.get(slot, 0) + 1
                if consec[slot] == 3:
                    total_violations += 1
                    print(f"[VIOLATION] #{total_violations}  "
                          f"face {slot+1}  conf={conf*100:.0f}%")
                    if save_violations:
                        save_screenshot(frame.copy(), total_violations)

        draw_hud(frame, len(faces), with_mask_count, without_mask_count, fps)

        fc += 1
        el  = time.time() - fps_t
        if el >= 1.0:
            fps = fc / el; fc = 0; fps_t = time.time()

        cv2.imshow("Face Mask Detection", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("s"):
            save_screenshot(frame.copy(), total_violations)
        elif k == ord("d"):
            debug = not debug
            print(f"[INFO]  Debug mode: {'ON' if debug else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE]  Total violations (no mask): {total_violations}")

# ─── Entry ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",  default="webcam")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--debug",   action="store_true",
                    help="Print raw prediction scores per face")
    args = ap.parse_args()
    run(source=args.source,
        save_violations=not args.no_save,
        debug=args.debug)
