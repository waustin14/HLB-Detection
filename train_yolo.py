from ultralytics.models import YOLO
from YOLOWeightedDataset import YOLOWeightedDataset
import ultralytics.data.build as build

# Set the YOLOWeightedDataset as the dataset for the YOLO model
build.YOLODataset = YOLOWeightedDataset

# Load the base YOLO model
model = YOLO("yolo11m.pt")

model.train(
    data="hlb-detection-v1.yaml",     # your dataset config
    epochs=100,
    imgsz=640,                    # image size
    batch=64,                     # adjust depending on GPU VRAM
    device=0,                     # GPU ID
    workers=8,                    # number of dataloader workers
    optimizer="SGD",              # or "AdamW"
    lr0=0.01,                     # initial learning rate
    lrf=0.01,                     # final learning rate (lr0 * lrf)
    momentum=0.937,               # default momentum for SGD
    weight_decay=0.0005,
    dropout=0.0,
    pretrained=True,             # use pretrained weights
    save=True,
    save_period=10,              # save every N epochs
    project="hlb-training",
    name="yolov8m-l40s-20k",
    exist_ok=True,               # overwrite if the run dir exists
    patience=20,                 # early stopping patience
    val=True,                    # run validation during training
    verbose=True
)
