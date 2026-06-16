"""
Face Mask Detection - Training Script
MobileNetV2 transfer learning: with_mask / without_mask
Dataset layout:
    dataset/
        with_mask/     ← face images WITH mask
        without_mask/  ← face images WITHOUT mask
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 1e-4
DATASET_DIR   = "dataset"
MODEL_SAVE    = "models/mask_detector.keras"
CLASSES       = ["with_mask", "without_mask"]

os.makedirs("models", exist_ok=True)

# ─── Load images ──────────────────────────────────────────────────────────────
def load_data():
    print("[INFO]  Loading images from dataset...")
    data   = []
    labels = []
    exts   = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls in CLASSES:
        cls_path = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_path):
            print(f"[WARN]  Folder not found: {cls_path}")
            continue
        files = [f for f in os.listdir(cls_path)
                 if os.path.splitext(f)[1].lower() in exts]
        print(f"[INFO]    {cls}: {len(files)} images")
        for fname in files:
            fpath = os.path.join(cls_path, fname)
            img   = load_img(fpath, target_size=IMG_SIZE)
            arr   = img_to_array(img)
            arr   = preprocess_input(arr)
            data.append(arr)
            labels.append(cls)

    data   = np.array(data,  dtype="float32")
    labels = np.array(labels)

    lb     = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = tf.keras.utils.to_categorical(labels, 2)

    print(f"[INFO]  Total images: {len(data)}")
    return data, labels

# ─── Build model ──────────────────────────────────────────────────────────────
def build_model():
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=Input(shape=(*IMG_SIZE, 3))
    )
    base.trainable = False   # freeze base

    head = base.output
    head = AveragePooling2D(pool_size=(7, 7))(head)
    head = Flatten()(head)
    head = Dense(128, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(2,   activation="softmax")(head)

    model = Model(inputs=base.input, outputs=head)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ─── Train ────────────────────────────────────────────────────────────────────
def train():
    data, labels = load_data()
    if len(data) == 0:
        print("[ERROR] No images found. Check dataset/ folder.")
        return

    # Split 80% train / 20% test
    (X_train, X_test, y_train, y_test) = train_test_split(
        data, labels, test_size=0.20, stratify=labels, random_state=42
    )

    # Augmentation
    aug = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    print("\n[INFO]  Building model...")
    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                          patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\n[INFO]  Training...")
    H = model.fit(
        aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Evaluate
    print("\n[INFO]  Evaluating...")
    pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = np.argmax(y_test, axis=1)
    print(classification_report(true_labels, pred_labels,
                                 target_names=CLASSES))

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1,2,1)
    plt.plot(H.history["accuracy"],     label="Train acc")
    plt.plot(H.history["val_accuracy"], label="Val acc")
    plt.title("Accuracy"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(H.history["loss"],     label="Train loss")
    plt.plot(H.history["val_loss"], label="Val loss")
    plt.title("Loss"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("models/training_plot.png")
    print(f"\n[DONE]  Model saved → {MODEL_SAVE}")
    print("[DONE]  Plot saved  → models/training_plot.png")

if __name__ == "__main__":
    train()
