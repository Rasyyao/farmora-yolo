from ultralytics import YOLO

model = YOLO("runs/detect/farmora_gulma-4/weights/best.pt")  # ganti ke v3
results = model.predict(
    "datasets/merged/real_test_outside_dataset/rumput-teki-1.jpg",
    save=True,
    conf=0.28,      # turunin sesuai threshold final yang kita putusin (Amaranthus lebih longgar)
    iou=0.4,         # turunin dari 0.55, biar box numpuk lebih ke-filter
    imgsz=960,       # penting! v3 dilatih di 960, sebaiknya predict juga di resolusi sama
    augment=True
)

for box in results[0].boxes:
    print(model.names[int(box.cls)], float(box.conf))