from ultralytics import YOLO

# Load YOLOv8 pretrained model
model = YOLO("yolov8n.pt")  # You can replace with yolov8s.pt if needed

def detect_fashion_items(image_path):
    results = model(image_path)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            bbox = box.xywh[0].tolist()  # [x_center, y_center, width, height]
            confidence = float(box.conf[0])
            detections.append({
                "class": cls_name,
                "bbox": bbox,
                "confidence": confidence
            })
    return detections
