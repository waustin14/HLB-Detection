from ultralytics.models import YOLO
from YOLOWeightedDataset import YOLOWeightedDataset
import ultralytics.data.build as build

# Set the YOLOWeightedDataset as the dataset for the YOLO model
build.YOLODataset = YOLOWeightedDataset

# Load the base YOLO model
model = YOLO("yolov11n.pt")  # Load a pre-trained YOLOv11 model

# Set the model to training mode
model.train(
    data="hlb-detection-v1.yaml",
    epochs=500,
    batch=16,
    imgsz=640,
    device="cuda",
    workers=4
)