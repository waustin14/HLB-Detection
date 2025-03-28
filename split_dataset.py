import os
import random
import shutil
from pathlib import Path

# Settings
SOURCE_IMAGE_DIR = Path("./images")
SOURCE_LABEL_DIR = Path("./labels")
TARGET_IMAGE_DIR = Path("./images")
TARGET_LABEL_DIR = Path("./labels")

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.2,
    "test": 0.1
}

IMG_EXTENSIONS = [".jpg", ".jpeg", ".png"]  # Add more if needed

# Ensure subdirectories exist
for split in SPLIT_RATIO.keys():
    (TARGET_IMAGE_DIR / split).mkdir(parents=True, exist_ok=True)
    (TARGET_LABEL_DIR / split).mkdir(parents=True, exist_ok=True)

# Get all image files
image_files = [f for f in SOURCE_IMAGE_DIR.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]

# Shuffle images
random.shuffle(image_files)

# Calculate split sizes
total = len(image_files)
train_end = int(SPLIT_RATIO["train"] * total)
val_end = train_end + int(SPLIT_RATIO["val"] * total)

splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# Move files
for split_name, files in splits.items():
    for img_path in files:
        label_path = SOURCE_LABEL_DIR / img_path.with_suffix('.txt').name

        # Copy image
        shutil.copy(img_path, TARGET_IMAGE_DIR / split_name / img_path.name)

        # Copy label if it exists
        if label_path.exists():
            shutil.copy(label_path, TARGET_LABEL_DIR / split_name / label_path.name)
        else:
            print(f"Warning: Label file not found for {img_path.name}")

print("✅ Dataset split completed.")
