import os
import shutil
import random

DATASET_DIR = "yolo_dataset"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "train", "images")
TRAIN_LABELS = os.path.join(DATASET_DIR, "train", "labels")
VAL_IMAGES = os.path.join(DATASET_DIR, "valid", "images")
VAL_LABELS = os.path.join(DATASET_DIR, "valid", "labels")

VAL_RATIO = 0.2
SEED = 42

os.makedirs(VAL_IMAGES, exist_ok=True)
os.makedirs(VAL_LABELS, exist_ok=True)

images = [f for f in os.listdir(TRAIN_IMAGES) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.seed(SEED)
random.shuffle(images)

val_count = int(len(images) * VAL_RATIO)
val_files = images[:val_count]
train_files = images[val_count:]

for fname in val_files:
    stem = os.path.splitext(fname)[0]
    shutil.move(os.path.join(TRAIN_IMAGES, fname), os.path.join(VAL_IMAGES, fname))
    label_file = stem + ".txt"
    label_src = os.path.join(TRAIN_LABELS, label_file)
    if os.path.exists(label_src):
        shutil.move(label_src, os.path.join(VAL_LABELS, label_file))

print(f"Total  : {len(images)}")
print(f"Train  : {len(train_files)} ({len(train_files)/len(images)*100:.1f}%)")
print(f"Val    : {len(val_files)} ({len(val_files)/len(images)*100:.1f}%)")
print("Done.")
