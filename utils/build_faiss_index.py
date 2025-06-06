# utils/build_faiss_index.py

import pandas as pd
import requests
from PIL import Image
import numpy as np
import faiss
from models.clip_matcher import embed_image
import os

def build_index(csv_path="data/catalog.csv"):
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_cols = ["id", "image_url"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f" Column '{col}' not found in CSV. Found: {list(df.columns)}")

    # Only use the first image per product ID
    df_unique = df.drop_duplicates(subset=["id"], keep="first")

    product_ids = []
    embeddings = []

    for _, row in df_unique.iterrows():
        pid = str(row["id"])
        image_url = row["image_url"]
        try:
            response = requests.get(image_url, stream=True, timeout=5)
            response.raise_for_status()
            img = Image.open(response.raw).convert("RGB")
            emb = embed_image(img)
            embeddings.append(emb)
            product_ids.append(pid)
        except Exception as e:
            print(f" Skipping ID {pid} due to error: {e}")

    if not embeddings:
        raise RuntimeError(" No valid embeddings generated. Check your image URLs.")

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    os.makedirs("data", exist_ok=True)
    faiss.write_index(index, "data/faiss_index.bin")
    with open("data/product_ids.txt", "w") as f:
        f.write("\n".join(product_ids))

    print(f" FAISS index built and saved. Indexed {len(product_ids)} products.")

if __name__ == "__main__":
    build_index("data/catalog.csv")
