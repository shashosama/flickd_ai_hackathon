from ultralytics import YOLO #import the yolo class from the ultralytics library 

# Load the YOLO model for deepfashion detection
model = YOLO("models/yolov8_deepfashion.pt")
#define a set of target fashion-related classes to filter the detections
FASHION_CLASSES = {
    "person","dress", "handbag", "backpack", "tie", "suitcase", "umbrella",
    "hat", "jacket", "coat", "scarf", "pants", "shirt", "blouse", "shorts", "skirt", "jeans", "shoe", "sneaker", "boot"
}

#define a function to detect and filter fashion items in an input image 
def detect_fashion_items(image_path):
    #run the yolo model on image 
    results = model(image_path)
    #initialize list to hold filtered detections 
    detections = []
#iterate through all result objects 
    for r in results:
        #iterate through all detected bounding boxes in the result 
        for box in r.boxes:
           #get the class index  as a integer 
            cls_idx = int(box.cls[0])
            #convert the class index to the class name using the model's label map 
            cls_name = model.names[cls_idx]
            print(f"Detected class: {cls_name}")
            #if the class is not in the predefined fashion classes , skip it 


            if cls_name.lower() not in FASHION_CLASSES:
                continue
            #get the bounding box as a list of floats 

            bbox = box.xywh[0].tolist()
            #get the confidence score of the detection
            confidence = float(box.conf[0])
            #append the relevent detection info to the list 

            detections.append({
                "class": cls_name,
                "bbox": bbox,
                "confidence": confidence
            })
            #log the final iltered detections 

    print(f"Filtered detections: {detections}")
    #return the list of filtered detection dictionaries 
    return detections
