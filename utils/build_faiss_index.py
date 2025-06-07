import pandas as pd
import requests
from PIL import Image
import numpy as np
import faiss
from models.clip_matcher import embed_image

def build_index(csv_path="data/product_catalog.csv"):
    df = pd.read_csv(csv_path)
    product_ids = df["product_id"].tolist()
    embeddings = []

    for _, row in df.iterrows():
        try:
            img = Image.open(requests.get(row["image_url"], stream=True).raw).convert("RGB")
            emb = embed_image(img)
            embeddings.append(emb)
        except Exception as e:
            print(f"[ERROR] Skipping {row['product_id']}: {e}")

    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "data/faiss_index.bin")
    with open("data/product_ids.txt", "w") as f:
        f.write("\n".join(product_ids))

    print("[DONE] FAISS index built and saved.")
