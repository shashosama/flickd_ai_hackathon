import pandas as pd                              #for reading CSV data
import requests                                  #for downloading images from URL 
from io import BytesIO                           #for handling in-mi=emory binary streams 
from PIL import Image                            #for opening and processing images
import torch                                     #for using Pytorch - based CLIP model 
import numpy as np                               #for numerical operations and array hand 
import faiss                                     #Facebook AI similarity search 
from transformers import CLIPProcessor, CLIPModel#Hugging face CLip model 
import os                                        #for file and directory operations 
import pickle                                    #for saving python objects to disk 

# Load CLIP modeland its associated processor from Hugging FAce 
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
#downloads image from the shopify URL 
def download_image(url):
    try:
        response = requests.get(url, timeout=5)                     #try to fetch the image withing 5 secs 
        return Image.open(BytesIO(response.content)).convert("RGB") #convertto PIL image in RGB format 
    except:
        return None                                                 # return none if download or consversion fails 
#Generates a CLIP embedding (vector) for a given image 
def get_clip_embedding(image):                                      #preprocess image for CLIP input 
    inputs = clip_processor(images=image, return_tensors="pt")      #Preprocess image for CLIP input 
with torch.no_grad():                                           #Disable gradient tracking for inference
    embedding = clip_model.get_image_features(**inputs)             #extract image embedding using CLIP 
return embedding[0].cpu().numpy()                               #convert tensor to numpy array and return

#main function to build the FAISS index from the product catalog CSV
def build_faiss_index(csv_path, output_dir="faiss_index"):
    os.makedirs(output_dir, exist_ok=True)  #create the output folder if it doesnt exist 
    df = pd.read_csv(csv_path)              #read the product catalog CSV into a dataframe
    embeddings = []                         #lists to store call image embeddings
    ids = []                                #list to store corresponding product IDs 

    for i, row in df.iterrows():            #Iterate over each product row in the CSV 
        image = download_image(row["Shopify image URL"]) #Download the product imaage
        if image is None:                               #skip the product if image failed to load 
            continue                                   
        emb = get_clip_embedding(image)                 #Get the image's CLIP embedding
        embeddings.append(emb)                          #Store embedding in list 
        ids.append(row["Product ID"])                   #Store corresponding product ID

    emb_matrix = np.vstack(embeddings).astype("float32")#Stack all embeddings into one Numpy matrix

    # Normalize embeddings (recommended for cosine similarity)
    faiss.normalize_L2(emb_matrix)

    index = faiss.IndexFlatIP(emb_matrix.shape[1])  #Initalize FAISS index 
    index.add(emb_matrix)                           #adds all normalized embeddings to the index

    # Save index and ID map
    faiss.write_index(index, os.path.join(output_dir, "clip_index.faiss"))
    with open(os.path.join(output_dir, "product_ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    print(f"Indexed {len(ids)} products.")#final message showing how many products were sucessfully indexed 
    #build_faiss_index("data/catalog.csv")