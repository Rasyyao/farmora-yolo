from ultralytics import YOLO
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

model = YOLO(str(SCRIPT_DIR / "yolov8s.pt"))

results = model.train(
    data=str(SCRIPT_DIR / "data.yaml"),
    epochs=80,
    imgsz=960,
    device="mps",
    batch=8,
    patience=20,
    name="farmora_gulma"
)