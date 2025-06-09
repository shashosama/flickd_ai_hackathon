import numpy as np # importing a numpy 
import faiss# importing faiss 
from transformers import CLIPProcessor, CLIPModel #importing clip model and processor 
from PIL import Image#importing pyTorch
import torch

# Load model + processor once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") #load the pre-trained CLIP model from Hugging face 
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")#Loading the corresponding processor which prepares images for the clip model

def embed_image(image: Image.Image): #define a function to convert an input image to normalized CLIP embedding 
    inputs = clip_processor(images=image, return_tensors="pt") #process the image into tensors suitable for the clip model 
    with torch.no_grad():# using the clip model to get the image embedding is shapped correctly as a @D array and covert to float32 
        embedding = clip_model.get_image_features(**inputs)
    embedding = embedding.numpy().astype("float32")#convert the embedding tensor to a numpy array of float32
    faiss.normalize_L2(embedding)#normalize the embedding using L2 norm
    return embedding[0]  # return flat array

def find_best_match(embedding, index, product_ids):#define a function to find the most similar product from a FAISS index 
    embedding = np.array(embedding).reshape(1, -1).astype("float32")#ensure the embedding is shapped correctly as a 2d array and covert to float32 
    faiss.normalize_L2(embedding)#normalizing the query embedding to L2 norm 
    D, I = index.search(embedding, 1)#use faiss to search for the top -1 closest vector in the index 
    score = float(D[0][0])#extract the similarity score of the top match 
    idx = int(I[0][0])#extract the index of the top matched item 
    #determine the type of match based on similarity thresholds
    match_type = (
        "exact" if score > 0.9 else#very close 
        "similar" if score > 0.75 else#moderately close match 
        "no match"# below similarity threshold 
    )
    #return the match type, matched product ID and similarity score 
    return match_type, product_ids[idx], score
