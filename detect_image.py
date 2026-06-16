"""
Face Mask Detection - Single Image Detection with Screenshot
Usage:
    py detect_image.py --image photo.jpg              # show result
    py detect_image.py --image photo.jpg --save       # auto save result
    py detect_image.py --image photo.jpg --screenshot # save only violations
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import argparse
import os
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH     = "models/mask_detector.keras"
FACE_PROTO     = "face_detector/deploy.prototxt"
FACE_WEIGHTS   = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
IMG_SIZE       = (224, 224)
CONF_THR       = 0.50
MASK_THR       = 0.65
COLOR_MASK     = (0, 200, 80)
COLOR_NO_MASK  = (0, 60, 220)
SCREENSHOTS_DIR = "screenshots"

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

# ─── Load models ──────────────────────────────────────────────────────────────
def load_models():
    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_WEIGHTS):
        print("[ERROR] Face detector files missing in face_detector/ folder.")
        exit(1)
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}.")
        print("        Run py train_mask_detector.py first.")
        exit(1)
    face_net = cv2.dnn.readNet(FACE_PROTO, FACE_WEIGHTS)
    mask_net = load_model(MODEL_PATH)
    print("[INFO]  Models loaded.\n")
    return face_net, mask_net

# ─── Save screenshot ──────────────────────────────────────────────────────────
def save_screenshot(frame, label="result"):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SCREENSHOTS_DIR, f"{label}_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[SAVED] Screenshot → {path}")
    return path

# ─── Detect and annotate ──────────────────────────────────────────────────────
def detect(image_path, face_net, mask_net):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        exit(1)

    h, w = frame.shape[:2]
    print(f"[INFO]  Image: {image_path}  ({w}x{h})")

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,177.0,123.0))
    face_net.setInput(blob)
    dets = face_net.forward()

    mask_count    = 0
    no_mask_count = 0
    total_faces   = 0

    for i in range(dets.shape[2]):
        conf = float(dets[0,0,i,2])
        if conf < CONF_THR:
            continue

        box = dets[0,0,i,3:7] * np.array([w,h,w,h])
        x1,y1,x2,y2 = box.astype(int)
        x1=max(0,x1); y1=max(0,y1)
        x2=min(w-1,x2); y2=min(h-1,y2)
        if x2-x1 < 20 or y2-y1 < 20:
            continue

        total_faces += 1

        crop  = frame[y1:y2, x1:x2]
        face  = cv2.resize(crop, IMG_SIZE)
        face  = preprocess_input(img_to_array(face).astype("float32"))
        preds = mask_net.predict(np.expand_dims(face,0), verbose=0)[0]
        with_p    = float(preds[0])
        without_p = float(preds[1])
        max_p     = max(with_p, without_p)

        if max_p < MASK_THR:
            label = "Uncertain"
            color = (150,150,150)
        elif with_p > without_p:
            label = "Mask"
            color = COLOR_MASK
            mask_count += 1
        else:
            label = "No Mask"
            color = COLOR_NO_MASK
            no_mask_count += 1

        print(f"  Face {total_faces}: {label:10}  "
              f"with_mask={with_p:.2f}  without_mask={without_p:.2f}")

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        text  = f"{label} {max_p*100:.0f}%"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(0.9, w/800))
        (tw,th),_ = cv2.getTextSize(text,font,scale,1)
        ly = max(y1-th-10,0)
        cv2.rectangle(frame,(x1,ly),(x1+tw+10,ly+th+10),color,-1)
        cv2.putText(frame,text,(x1+5,ly+th+5),font,scale,(255,255,255),1,cv2.LINE_AA)

    # Summary bar at bottom
    summary = (f"Faces: {total_faces}  |  "
               f"Mask: {mask_count}  |  "
               f"No Mask: {no_mask_count}")
    cv2.rectangle(frame,(0,h-40),(w,h),(20,20,20),-1)
    cv2.putText(frame, summary, (10, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1, cv2.LINE_AA)

    # Timestamp watermark
    ts_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts_text, (w-200, h-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1, cv2.LINE_AA)

    # Terminal summary
    print(f"\n  ── Summary ──────────────────")
    print(f"  Total Faces : {total_faces}")
    print(f"  With Mask   : {mask_count}  ✅")
    print(f"  No Mask     : {no_mask_count}  ❌")
    if no_mask_count > 0:
        print(f"\n  ⚠️  VIOLATION: {no_mask_count} person(s) not wearing mask!")
    else:
        print(f"\n  ✅  All persons are wearing masks.")

    return frame, mask_count, no_mask_count

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Face Mask Detection on Image")
    ap.add_argument("--image",      required=True,
                    help="Path to input image e.g. photo.jpg")
    ap.add_argument("--save",       action="store_true",
                    help="Save annotated result always")
    ap.add_argument("--screenshot", action="store_true",
                    help="Save screenshot only when violation (No Mask) found")
    args = ap.parse_args()

    face_net, mask_net = load_models()
    result, mask_count, no_mask_count = detect(args.image, face_net, mask_net)

    # ── Screenshot logic ──────────────────────────────────────────────────────
    if args.save:
        # Always save result
        save_screenshot(result, label="result")

    elif args.screenshot:
        # Save only if there is a violation
        if no_mask_count > 0:
            save_screenshot(result, label="violation")
        else:
            print("[INFO]  No violation found — screenshot not saved.")

    # ── Show result window ────────────────────────────────────────────────────
    print("\n[INFO]  Press  S  to save screenshot  |  Any other key to close")
    cv2.imshow("Mask Detection Result", result)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("s") or key == ord("S"):
            save_screenshot(result, label="manual")
            print("[INFO]  Press any other key to close.")
        else:
            break

    cv2.destroyAllWindows()
