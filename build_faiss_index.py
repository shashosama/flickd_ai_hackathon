import pandas as pd                         #for reading the product catalog CSV 
import requests                             #for  downloading images from the internet
from PIL import Image                       #for opening and converting image files 
import numpy as np                          #for working with arrays of numbers
import faiss                                #fast library to search for similar images 
from models.clip_matcher import embed_image # Function to turn image into a CLIP vector

def build_index(csv_path="data/product_catalog.csv"):
    df = pd.read_csv(csv_path)                      #define a function to create a FAISS index from the product catalog CSV 
    product_ids = df["product_id"].tolist()         #Get a list of all product IDS
    embeddings = []                                 #This will store the number-code for each image

    for i, row in df.iterrows():                    #Go through each product in the catalog  
        try:
            img = Image.open(requests.get(row["image_url"], stream=True).raw).convert("RGB")
            #Download the product image from the internet
            emb = embed_image(img)
            #turn the image into a sepcial set numbers 
            embeddings.append(emb)
            #add this images code to our basket 
        except Exception as e:
            print(f"Error embedding {row['product_id']}: {e}") #print error if something goes wrong

    embeddings = np.array(embeddings).astype("float32")        #turn list of codes into a big number grid
    index = faiss.IndexFlatL2(embeddings.shape[1])             #create a FAISS index using euclidean distance 
    index.add(embeddings)                                      #add all image codes to the smart search box 

    faiss.write_index(index, "data/faiss_index.bin")           #save the smart search box to a file 
    with open("data/product_ids.txt", "w") as f:               #save the product ID list
        f.write("\n".join(product_ids))                        #Print debugging message

    print("FAISS index built and saved.")
