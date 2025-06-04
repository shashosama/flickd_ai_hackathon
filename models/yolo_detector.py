from ultralytics import YOLO
#import yolo model 


model = YOLO("yolov8n.pt")              #loading a pretained model that knows how to detect things like jackets and backets 

def detect_fashion_items(image_path):   #Writimg function that takes an image file and looks for clothes or accessories 
    results = model(image_path)         #Giving the image to YOLO and it returns a bunch of box with guesses of what is in the image
    detections = []                     #Preparing a empty list 
    for r in results:                   #iterate through each result (per image)
        for box in r.boxes:             #Get predicted class name using the class index
            cls_name = model.names[int(box.cls)]        
            bbox = box.xywh[0].tolist()              # extract the bound in box format [x_center, y_center, width, height]
            confidence = float(box.conf[0])          #getting the confidence score (probabilty) of detection
            detections.append({                      #append the detection details to the output list 
                "class": cls_name,                   #detected class label   
                "bbox": bbox,                        #Bounding box coordinates
                "confidence": confidence             #Confidence score 
            })
    return detections                                #Returns the full list of detected objects with their class labels, bounding boxes, and confidence scores.
