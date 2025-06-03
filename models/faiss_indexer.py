import pandas as pd
import requests
from io import BytesIO
from PIL import Image
import torch
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel
import os
import pickle

# Load CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        return Image.open(BytesIO(response.content)).convert("RGB")
    except:
        return None

def get_clip_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)
    return embedding[0].cpu().numpy()

def build_faiss_index(csv_path, output_dir="faiss_index"):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    embeddings = []
    ids = []

    for i, row in df.iterrows():
        image = download_image(row["Shopify image URL"])
        if image is None:
            continue
        emb = get_clip_embedding(image)
        embeddings.append(emb)
        ids.append(row["Product ID"])

    emb_matrix = np.vstack(embeddings).astype("float32")

    # Normalize embeddings (recommended for cosine similarity)
    faiss.normalize_L2(emb_matrix)

    index = faiss.IndexFlatIP(emb_matrix.shape[1])
    index.add(emb_matrix)

    # Save index and ID map
    faiss.write_index(index, os.path.join(output_dir, "clip_index.faiss"))
    with open(os.path.join(output_dir, "product_ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    print(f"Indexed {len(ids)} products.")
