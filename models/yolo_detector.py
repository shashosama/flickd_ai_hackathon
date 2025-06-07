from ultralytics import YOLO

# Load the YOLO model once
model = YOLO("models/yolov8_deepfashion.pt")

FASHION_CLASSES = {
    "person","dress", "handbag", "backpack", "tie", "suitcase", "umbrella",
    "hat", "jacket", "coat", "scarf", "pants", "shirt", "blouse", "shorts", "skirt", "jeans", "shoe", "sneaker", "boot"
}

def detect_fashion_items(image_path):
    results = model(image_path)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            cls_name = model.names[cls_idx]
            print(f"Detected class: {cls_name}")

            if cls_name.lower() not in FASHION_CLASSES:
                continue

            bbox = box.xywh[0].tolist()
            confidence = float(box.conf[0])

            detections.append({
                "class": cls_name,
                "bbox": bbox,
                "confidence": confidence
            })

    print(f"Filtered detections: {detections}")
    return detections
