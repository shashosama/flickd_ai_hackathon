from ultralytics import YOLO

# Load YOLOv8 model (can be replaced with fine-tuned DeepFashion model later)
model = YOLO("yolov8n.pt")

# Define relevant fashion classes (adjust as needed based on your YOLO model's output)
FASHION_CLASSES = {
    "person", "dress", "handbag", "backpack", "tie", "suitcase", "umbrella",
    "hat", "jacket", "coat", "scarf", "pants", "shirt", "blouse", "shorts", "skirt", "jeans", "shoe", "sneaker", "boot"
}

def detect_fashion_items(image_path):
    """
    Detect fashion items in an image using YOLOv8.

    Args:
        image_path (str): Path to the image file.

    Returns:
        List[dict]: List of detections with class, bbox, and confidence.
    """
    results = model(image_path)  # Run YOLO inference on the image
    detections = []

    for r in results:
        for box in r.boxes:
            cls_idx = int(box.cls[0])
            cls_name = model.names[cls_idx]

            # Filter to only fashion-related classes
            if cls_name.lower() not in FASHION_CLASSES:
                continue

            bbox = box.xywh[0].tolist()  # [x_center, y_center, width, height]
            confidence = float(box.conf[0])

            detections.append({
                "class": cls_name,
                "bbox": bbox,
                "confidence": confidence
            })

    return detections
