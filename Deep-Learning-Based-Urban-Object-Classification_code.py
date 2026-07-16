"""
=============================================================================
EPM106 Machine Learning — Group Project 2
Optimised version:
- Skips YOLO/cropping if cropped_dataset already exists
- Keeps cropped images as PNG
- Uses tf.data instead of ImageDataGenerator
- Uses MobileNetV2 instead of ResNet50
- Uses fast lightweight Custom CNN
- Enables cache + prefetch
- CNN trains for 10 epochs
- Transfer model trains for 20 epochs per phase
- Makes learning-rate sweep optional
- Saves and shows plots
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

print("TensorFlow version :", tf.__version__)
print("GPU available      :", tf.config.list_physical_devices("GPU"))

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = BASE_DIR

OUT_DIR = os.path.join(BASE_DIR, "cropped_dataset")
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

IMG_SIZE = (128, 128)
BATCH_SIZE = 64

CLASSES = ["car", "pedestrian", "bike"]
NUM_CLASSES = len(CLASSES)

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test"
}

YOLO_TO_CLASS = {
    0: "pedestrian",
    1: "bike",
    3: "bike",
}

CNN_EPOCHS = 20
TRANSFER_EPOCHS = 20

RUN_LR_SWEEP = True
SHOW_PLOTS = True

AUTOTUNE = tf.data.AUTOTUNE

for split_out in ["train", "val", "test"]:
    for cls in CLASSES:
        os.makedirs(os.path.join(OUT_DIR, split_out, cls), exist_ok=True)

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("\nDataset root :", DATASET_ROOT)
print("Output dir   :", OUT_DIR)
print("Plots dir    :", PLOTS_DIR)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_and_show_plot(filename):
    """
    Saves the current Matplotlib figure into the plots folder, optionally displays
    it on screen, and then closes it to avoid memory usage building up during the
    full experiment.
    """
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def cropped_dataset_exists():
    """
    Checks whether cropped object images already exist in the cropped_dataset
    folder. This prevents the script from running YOLO and cropping again every
    time the program is executed.
    """
    total = 0

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            cls_dir = Path(OUT_DIR) / split / cls
            total += len(list(cls_dir.glob("*.png")))

    return total > 0


def count_existing_crops():
    """
    Counts how many cropped PNG images already exist for each class and split.
    This is used when crop generation is skipped so the dataset summary can still
    be printed correctly.
    """
    counters = defaultdict(lambda: defaultdict(int))

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            cls_dir = Path(OUT_DIR) / split / cls
            counters[split][cls] = len(list(cls_dir.glob("*.png")))

    return counters

    
def parse_voc_xml(xml_path):
    """
    Reads a Pascal VOC XML annotation file and extracts object labels and bounding
    box coordinates. It returns a list of objects together with the original image
    width and height.
    """
    
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")

    if size_el is None:
        return [], 0, 0

    width_el = size_el.find("width")
    height_el = size_el.find("height")

    if width_el is None or height_el is None:
        return [], 0, 0

    img_w = int(float(width_el.text))
    img_h = int(float(height_el.text))

    objects = []

    for obj in root.findall("object"):
        name_el = obj.find("name")

        if name_el is None:
            name_el = obj.find("n")

        if name_el is None or name_el.text is None:
            continue

        label = name_el.text.strip().lower()
        bb = obj.find("bndbox")

        if bb is None:
            continue

        xmin_el = bb.find("xmin")
        ymin_el = bb.find("ymin")
        xmax_el = bb.find("xmax")
        ymax_el = bb.find("ymax")

        if None in [xmin_el, ymin_el, xmax_el, ymax_el]:
            continue

        xmin = max(0, int(float(xmin_el.text)))
        ymin = max(0, int(float(ymin_el.text)))
        xmax = min(img_w, int(float(xmax_el.text)))
        ymax = min(img_h, int(float(ymax_el.text)))

        if xmax > xmin and ymax > ymin:
            objects.append((label, xmin, ymin, xmax, ymax))

    return objects, img_w, img_h


def save_crop(img_bgr, x1, y1, x2, y2, out_path, pad=8):
    """
    Crops an object region from the original image using the bounding box,
    adds small padding around the object, resizes the crop to IMG_SIZE, and saves
    it as a PNG image.
    """
    h, w = img_bgr.shape[:2]

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    crop = img_bgr[y1:y2, x1:x2]

    if crop.size == 0 or (x2 - x1) < 8 or (y2 - y1) < 8:
        return False

    crop_resized = cv2.resize(crop, IMG_SIZE)
    cv2.imwrite(out_path, crop_resized)

    return True


def find_matching_image(xml_path):
    """
    Finds the image file that matches a given XML annotation file by checking
    common image extensions such as .jpg, .jpeg, and .png.
    """
    for ext in [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
        candidate = xml_path.with_suffix(ext)

        if candidate.exists():
            return candidate

    return None


# =============================================================================
# TASKS 1-5 — DATA PROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("TASKS 1-5 — Data Process")
print("=" * 60)

if cropped_dataset_exists():
    print("Existing cropped_dataset found. Skipping YOLO detection and cropping.")
    counters = count_existing_crops()

else:
    print("No cropped_dataset found. Running XML crop extraction and YOLO detection.")

    from ultralytics import YOLO

    print("\nLoading YOLOv8n detector...")
    yolo = YOLO("yolov8n.pt")
    print("YOLOv8n ready.")

    counters = defaultdict(lambda: defaultdict(int))

    for robo_split, out_split in SPLIT_MAP.items():
        split_dir = os.path.join(DATASET_ROOT, robo_split)
        xml_files = sorted(Path(split_dir).glob("*.xml"))

        print(f"\n[{robo_split}] {len(xml_files)} XML files")

        for xml_path in tqdm(xml_files, desc=f"{robo_split}", ncols=72):
            img_path = find_matching_image(xml_path)

            if img_path is None:
                continue

            img_bgr = cv2.imread(str(img_path))

            if img_bgr is None:
                continue

            annotations, _, _ = parse_voc_xml(str(xml_path))

            for label, xmin, ymin, xmax, ymax in annotations:
                if label not in ("car", "vehicle"):
                    continue

                idx = counters[out_split]["car"]

                out_path = os.path.join(
                    OUT_DIR,
                    out_split,
                    "car",
                    f"car_{idx:05d}.png"
                )

                if save_crop(img_bgr, xmin, ymin, xmax, ymax, out_path):
                    counters[out_split]["car"] += 1

            results = yolo(
                str(img_path),
                verbose=False,
                conf=0.30,
                classes=list(YOLO_TO_CLASS.keys())
            )

            for result in results:
                for box in result.boxes:
                    cls = YOLO_TO_CLASS.get(int(box.cls))

                    if cls is None:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    idx = counters[out_split][cls]

                    out_path = os.path.join(
                        OUT_DIR,
                        out_split,
                        cls,
                        f"{cls}_{idx:05d}.png"
                    )

                    if save_crop(img_bgr, x1, y1, x2, y2, out_path):
                        counters[out_split][cls] += 1


print("\n" + "=" * 60)
print("TASK 5 — Dataset split summary")
print("=" * 60)

print(f"\n{'Class':12s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
print(f"{'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8}")

for cls in CLASSES:
    print(
        f"{cls:12s} "
        f"{counters['train'][cls]:8d} "
        f"{counters['val'][cls]:8d} "
        f"{counters['test'][cls]:8d}"
    )

# =============================================================================
# PLOT 1 — SAMPLE CROPS
# =============================================================================

print("\nPlotting sample crops...")

fig, axes = plt.subplots(len(CLASSES), 5, figsize=(14, 9))

fig.suptitle(
    "Tasks 1-3: Sample Extracted Sub-Images",
    fontsize=12,
    fontweight="bold"
)

for row, cls in enumerate(CLASSES):
    cls_dir = os.path.join(OUT_DIR, "train", cls)
    files = sorted(os.listdir(cls_dir))[:5] if os.path.exists(cls_dir) else []

    for col in range(5):
        ax = axes[row, col]

        if col < len(files):
            img_path = os.path.join(cls_dir, files[col])
            img = cv2.imread(img_path)

            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            if col == 0:
                ax.set_title(
                    cls.capitalize(),
                    fontsize=11,
                    fontweight="bold",
                    loc="left"
                )

        ax.axis("off")

save_and_show_plot("task1_sample_crops.png")

# =============================================================================
# PLOT 2 — CLASS DISTRIBUTION
# =============================================================================

print("\nPlotting class distribution...")

fig, ax = plt.subplots(figsize=(8, 5))

x = np.arange(len(CLASSES))
w = 0.25

for i, split in enumerate(["train", "val", "test"]):
    counts = [counters[split][cls] for cls in CLASSES]

    bars = ax.bar(
        x + i * w,
        counts,
        w,
        label=split.capitalize(),
        alpha=0.85
    )

    for bar in bars:
        ax.annotate(
            str(int(bar.get_height())),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8
        )

ax.set_xticks(x + w)
ax.set_xticklabels([c.capitalize() for c in CLASSES])
ax.set_ylabel("Number of images")
ax.set_title("Task 5 — Class Distribution", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)

save_and_show_plot("task5_class_distribution.png")

# =============================================================================
# TASK 4 & 5 — TF.DATA DATASETS
# =============================================================================

print("\n" + "=" * 60)
print("TASK 4 & 5 — Loading datasets with tf.data")
print("=" * 60)

train_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)

val_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "val"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

test_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(OUT_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)

class_names = train_ds_raw.class_names
print("Class names:", class_names)

normalisation = layers.Rescaling(1.0 / 255)

train_ds = (
    train_ds_raw
    .map(lambda x, y: (normalisation(x), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds_raw
    .map(lambda x, y: (normalisation(x), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds_raw
    .map(lambda x, y: (normalisation(x), y), num_parallel_calls=AUTOTUNE)
    .cache()
    .prefetch(AUTOTUNE)
)

y_true_test = np.concatenate([
    np.argmax(y.numpy(), axis=1)
    for _, y in test_ds_raw
])

# =============================================================================
# DATA AUGMENTATION FOR TRANSFER MODEL ONLY
# =============================================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.04),
        layers.RandomZoom(0.08),
    ],
    name="data_augmentation"
)

# =============================================================================
# TASK 6 — FAST CUSTOM CNN
# =============================================================================

def build_fast_custom_cnn(input_shape=(128, 128, 3), num_classes=3):
    """
    Builds a lightweight custom CNN from scratch. The model uses three
    convolutional blocks followed by global average pooling, a dense layer,
    dropout, and a softmax output layer for three-class classification.
    """
    inputs = keras.Input(shape=input_shape)

    # No augmentation here to keep Custom CNN fast
    x = inputs

    x = layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="FastCustomCNN")


print("\n" + "=" * 60)
print("TASK 6 — Fast Custom CNN")
print("=" * 60)

custom_cnn = build_fast_custom_cnn(
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES
)

custom_cnn.summary()

# =============================================================================
# TASK 7 — MOBILE NET TRANSFER MODEL
# =============================================================================

def build_transfer_model(input_shape=(128, 128, 3), num_classes=3):
    """
    Builds a transfer learning model using MobileNetV2 pre-trained on ImageNet.
    The MobileNetV2 backbone is frozen at first, and a new classification head is
    added for the project classes: car, pedestrian, and bike.
    """
    base = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    base.trainable = False

    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)

    x = layers.Lambda(
        lambda z: (z * 2.0) - 1.0,
        name="mobilenet_preprocess"
    )(x)

    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)

    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="MobileNetV2_Transfer")

    return model, base


print("\n" + "=" * 60)
print("TASK 7 — Transfer Learning model: MobileNetV2")
print("=" * 60)

transfer_model, mobilenet_base = build_transfer_model(
    input_shape=(*IMG_SIZE, 3),
    num_classes=NUM_CLASSES
)

transfer_model.summary()

# =============================================================================
# TASK 8 — TRAIN FAST CUSTOM CNN
# =============================================================================

print("\n" + "=" * 60)
print("TASK 8 — Training Fast Custom CNN")
print(f"Max epochs: {CNN_EPOCHS}")
print("=" * 60)

custom_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_cnn = custom_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CNN_EPOCHS,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "custom_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ],
    verbose=1
)

print("Fast Custom CNN training complete.")

# =============================================================================
# TASK 8 — TRAIN MOBILE NET PHASE 1
# =============================================================================

print("\n" + "=" * 60)
print("TASK 8 — Training MobileNetV2 Transfer: Phase 1")
print(f"Max epochs: {TRANSFER_EPOCHS}")
print("=" * 60)

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_p1 = transfer_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TRANSFER_EPOCHS,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ],
    verbose=1
)

print("MobileNetV2 Phase 1 complete.")

# =============================================================================
# TASK 8 — TRAIN MOBILE NET PHASE 2
# =============================================================================

print("\n" + "=" * 60)
print("TASK 8 — Training MobileNetV2 Transfer: Phase 2 fine-tuning")
print(f"Max epochs: {TRANSFER_EPOCHS}")
print("=" * 60)

mobilenet_base.trainable = True

for layer in mobilenet_base.layers[:-30]:
    layer.trainable = False

for layer in mobilenet_base.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_p2 = transfer_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=TRANSFER_EPOCHS,
    callbacks=[
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, "mobilenetv2_transfer_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ],
    verbose=1
)

print("MobileNetV2 Phase 2 complete.")

history_transfer = {
    k: history_p1.history[k] + history_p2.history[k]
    for k in history_p1.history
}

phase1_len = len(history_p1.history["loss"])

# =============================================================================
# TASK 9 — EVALUATION
# =============================================================================

def evaluate_model(model, val_ds, test_ds, y_true, class_names, model_name):
    """
    Evaluates a trained model on the validation and test datasets. It prints
    accuracy and loss, generates predictions for the test set, prints a
    classification report, and saves a confusion matrix plot.
    """
    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)

    print("\n" + "=" * 60)
    print(f"TASK 9 — Recognition Accuracy: {model_name}")
    print("=" * 60)
    print(f"Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"Validation Loss     : {val_loss:.4f}")
    print(f"Testing Accuracy    : {test_acc * 100:.2f}%")
    print(f"Testing Loss        : {test_loss:.4f}")

    y_prob = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"\nPer-class Classification Report — {model_name}:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
    else:
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Task 9 — Confusion Matrix: {model_name}", fontweight="bold")

    fname = model_name.lower().replace(" ", "_")
    save_and_show_plot(f"task9_confusion_{fname}.png")

    return y_pred, val_acc, test_acc


y_pred_cnn, val_acc_cnn, test_acc_cnn = evaluate_model(
    custom_cnn,
    val_ds,
    test_ds,
    y_true_test,
    class_names,
    "Fast Custom CNN"
)

y_pred_tl, val_acc_tl, test_acc_tl = evaluate_model(
    transfer_model,
    val_ds,
    test_ds,
    y_true_test,
    class_names,
    "MobileNetV2 Transfer"
)

# =============================================================================
# TASK 10 — TRAINING CURVES
# =============================================================================

print("\n" + "=" * 60)
print("TASK 10 — Plotting training curves")
print("=" * 60)

ep_cnn = range(1, len(history_cnn.history["loss"]) + 1)
ep_tl = range(1, len(history_transfer["loss"]) + 1)

fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 6))

fig.suptitle(
    "Task 10 — Training Processes: Fast Custom CNN vs MobileNetV2 Transfer",
    fontsize=13,
    fontweight="bold"
)

ax_loss.plot(ep_cnn, history_cnn.history["loss"], lw=2, label="Fast CNN — Train")
ax_loss.plot(ep_cnn, history_cnn.history["val_loss"], lw=2, ls="--", label="Fast CNN — Val")
ax_loss.plot(ep_tl, history_transfer["loss"], lw=2, label="MobileNetV2 — Train")
ax_loss.plot(ep_tl, history_transfer["val_loss"], lw=2, ls="--", label="MobileNetV2 — Val")
ax_loss.axvline(x=phase1_len, ls=":", lw=1.5, alpha=0.6, label="MobileNet Phase 1→2")
ax_loss.set_title("Loss vs Epochs", fontweight="bold")
ax_loss.set_xlabel("Epochs")
ax_loss.set_ylabel("Loss")
ax_loss.legend(fontsize=8)
ax_loss.grid(True, alpha=0.3)

ax_acc.plot(ep_cnn, [a * 100 for a in history_cnn.history["accuracy"]], lw=2, label="Fast CNN — Train")
ax_acc.plot(ep_cnn, [a * 100 for a in history_cnn.history["val_accuracy"]], lw=2, ls="--", label="Fast CNN — Val")
ax_acc.plot(ep_tl, [a * 100 for a in history_transfer["accuracy"]], lw=2, label="MobileNetV2 — Train")
ax_acc.plot(ep_tl, [a * 100 for a in history_transfer["val_accuracy"]], lw=2, ls="--", label="MobileNetV2 — Val")
ax_acc.axvline(x=phase1_len, ls=":", lw=1.5, alpha=0.6, label="MobileNet Phase 1→2")
ax_acc.set_title("Accuracy vs Epochs", fontweight="bold")
ax_acc.set_xlabel("Epochs")
ax_acc.set_ylabel("Accuracy (%)")
ax_acc.legend(fontsize=8)
ax_acc.grid(True, alpha=0.3)

save_and_show_plot("task10_training_curves.png")

# =============================================================================
# TASK 11 — PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 60)
print("TASK 11 — Compare performance")
print("=" * 60)

fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(2)
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    [val_acc_cnn * 100, test_acc_cnn * 100],
    width,
    label="Fast Custom CNN",
    alpha=0.85
)

bars2 = ax.bar(
    x + width / 2,
    [val_acc_tl * 100, test_acc_tl * 100],
    width,
    label="MobileNetV2 Transfer",
    alpha=0.85
)

for bar in list(bars1) + list(bars2):
    ax.annotate(
        f"{bar.get_height():.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
        xytext=(0, 5),
        textcoords="offset points",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

ax.set_title("Task 11 — Model Performance Comparison", fontweight="bold")
ax.set_ylabel("Recognition Accuracy (%)")
ax.set_xticks(x)
ax.set_xticklabels(["Validation Accuracy", "Testing Accuracy"])
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis="y", alpha=0.3)

save_and_show_plot("task11_comparison.png")

best_val_cnn = max(history_cnn.history["val_accuracy"]) * 100
best_val_tl = max(history_transfer["val_accuracy"]) * 100

winner = "MobileNetV2 Transfer" if test_acc_tl > test_acc_cnn else "Fast Custom CNN"

print(f"""
TASK 11 — Performance Summary
=======================================================
Model                     Best Val Acc     Test Acc
-------------------------------------------------------
Fast Custom CNN           {best_val_cnn:11.2f}% {test_acc_cnn * 100:9.2f}%
MobileNetV2 Transfer      {best_val_tl:11.2f}% {test_acc_tl * 100:9.2f}%
=======================================================

Winner: {winner}

Analysis:
- The Fast Custom CNN is built from scratch using three lightweight convolutional blocks.
- It removes heavy augmentation and reduces filter counts to improve speed.
- MobileNetV2 uses ImageNet pre-trained features, so it is usually faster to converge
  and more accurate on small datasets than a fully custom model.
- tf.data with cache and prefetch improves input loading speed.
- YOLO and cropping are skipped when cropped_dataset already exists.
""")

# =============================================================================
# TASK 12 — OPTIONAL LEARNING RATE SWEEP
# =============================================================================

if RUN_LR_SWEEP:
    print("\n" + "=" * 60)
    print("TASK 12 — Optional Learning Rate Sweep")
    print("=" * 60)

    lr_candidates = [1e-2, 1e-3, 1e-4]
    lr_results = {}

    for lr in lr_candidates:
        print(f"Training Fast Custom CNN with lr={lr}")

        m = build_fast_custom_cnn(
            input_shape=(*IMG_SIZE, 3),
            num_classes=NUM_CLASSES
        )

        m.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

        hist = m.fit(
            train_ds,
            validation_data=val_ds,
            epochs=6,
            verbose=1,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    patience=2,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
        )

        _, va = m.evaluate(val_ds, verbose=0)

        lr_results[lr] = {
            "val_acc": va,
            "history": hist.history
        }

        print(f"lr={lr} validation accuracy: {va * 100:.2f}%")

    best_lr = max(lr_results, key=lambda k: lr_results[k]["val_acc"])

    print(
        f"\nBest learning rate: {best_lr} "
        f"with validation accuracy {lr_results[best_lr]['val_acc'] * 100:.2f}%"
    )
    print("\nPlotting Task 12 — Learning Rate Sweep...")

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#FF5722", "#2196F3", "#4CAF50"]

    for color, (lr, data) in zip(colors, lr_results.items()):
        history = data["history"]
        epochs = range(1, len(history["loss"]) + 1)

        ax_loss.plot(epochs, history["loss"], color=color, lw=2, label=f"lr={lr} train")
        ax_loss.plot(epochs, history["val_loss"], color=color, lw=2, ls="--", label=f"lr={lr} val")

        ax_acc.plot(epochs, [a * 100 for a in history["accuracy"]],
                    color=color, lw=2, label=f"lr={lr} train")
        ax_acc.plot(epochs, [a * 100 for a in history["val_accuracy"]],
                    color=color, lw=2, ls="--", label=f"lr={lr} val")

    ax_loss.set_title("Loss vs Epochs", fontweight="bold")
    ax_loss.set_xlabel("Epochs")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_acc.set_title("Accuracy vs Epochs", fontweight="bold")
    ax_acc.set_xlabel("Epochs")
    ax_acc.set_ylabel("Accuracy (%)")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(True, alpha=0.3)

    save_and_show_plot("task12_lr_sweep.png")
    
    print("\nPlotting Task 12 — Final Accuracy Comparison...")

    lrs = list(lr_results.keys())
    accs = [lr_results[lr]["val_acc"] * 100 for lr in lrs]

    fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar([str(lr) for lr in lrs], accs, alpha=0.85)

    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title("Task 12 — Validation Accuracy vs Learning Rate", fontweight="bold")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.grid(axis="y", alpha=0.3)

    save_and_show_plot("task12_lr_bar.png")

    

else:
    print("\nTASK 12 — Learning-rate sweep skipped.")
    print("To enable it, set RUN_LR_SWEEP = True near the top of the script.")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("ALL TASKS COMPLETE")
print("=" * 60)

print(f"""
Task  1: Sub-images extracted or reused from cropped_dataset
Task  2: Three classes labelled — car, pedestrian, bike
Task  3: Crops stored as PNG in cropped_dataset/<split>/<class>/
Task  4: All crops resized to {IMG_SIZE[0]}x{IMG_SIZE[1]}
Task  5: Existing train / val / test split used
Task  6: Fast Custom CNN trained for up to {CNN_EPOCHS} epochs
Task  7: MobileNetV2 transfer model used instead of ResNet50
Task  8: Both models trained with validation
Task  9: Validation and test accuracy calculated
Task 10: Training curves saved
Task 11: Model performance compared
Task 12: Learning-rate sweep optional

Validation Accuracy:
  Fast Custom CNN     : {val_acc_cnn * 100:.2f}%
  MobileNetV2 Transfer: {val_acc_tl * 100:.2f}%

Testing Accuracy:
  Fast Custom CNN     : {test_acc_cnn * 100:.2f}%
  MobileNetV2 Transfer: {test_acc_tl * 100:.2f}%

Sub-images : {OUT_DIR}
Models     : {MODELS_DIR}
Plots      : {PLOTS_DIR}
""")

print("Output plots:")
for f in sorted(Path(PLOTS_DIR).glob("*.png")):
    print(f"  • {f.name}")

print("=" * 60)