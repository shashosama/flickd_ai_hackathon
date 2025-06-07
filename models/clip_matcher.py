import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model + processor once
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image: Image.Image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    embedding = embedding.numpy().astype("float32")
    faiss.normalize_L2(embedding)
    return embedding[0]  # return flat array

def find_best_match(embedding, index, product_ids):
    embedding = np.array(embedding).reshape(1, -1).astype("float32")
    faiss.normalize_L2(embedding)
    D, I = index.search(embedding, 1)
    score = float(D[0][0])
    idx = int(I[0][0])
    match_type = (
        "exact" if score > 0.9 else
        "similar" if score > 0.75 else
        "no match"
    )
    return match_type, product_ids[idx], score
