from PIL import Image
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel

# Load model & processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image: Image.Image) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)
    return embedding[0].cpu().numpy()

def find_best_match(query_emb: np.ndarray, index, product_ids: list, threshold_exact=0.9, threshold_similar=0.75):
    D, I = index.search(np.array([query_emb]), k=1)
    similarity = 1 - D[0][0]  # cosine distance to similarity
    matched_id = product_ids[I[0][0]]

    match_type = (
        "exact" if similarity > threshold_exact else
        "similar" if similarity > threshold_similar else
        "no match"
    )

    return match_type, matched_id, similarity
