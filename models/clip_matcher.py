from PIL import Image
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel

# Load model & processor
# loads openais CLIP model that turns images into embeddings 
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].cpu().numpy()
#converts a PIL image to CLIP vector 
#this vector is used for cosine similarity against your catalog 
def find_best_match(query_emb: np.ndarray, index, product_ids: list, threshold_exact=0.9, threshold_similar=0.75):
    #search for the nearest neighbor in FAISS index, D : squared cosine distances , I = index of matched item inyour produst_ids list 
    D, I = index.search(np.array([query_emb]), k=1)
    similarity = 1 - D[0][0]  # cosine distance to similarity
    #converts cosine distance to similarity (1- distance) 
    #retrieves the matched product ID 
    matched_id = product_ids[I[0][0]]
    #catagorizes result into: "exact match" , "similar match" , "no match"
    #exact = very clse #similar = somewhat close #no match = not close enough

    match_type = (
        "exact" if similarity > threshold_exact else
        "similar" if similarity > threshold_similar else
        "no match"
    )

    return match_type, matched_id, similarity
