# build_index.py

import os
from utils.build_faiss_index import build_index


def main():
    catalog_path = "data/catalog.csv"

    if not os.path.exists(catalog_path):
        print(f" Catalog file not found at: {catalog_path}")
        print("Make sure your product CSV exists and is named correctly.")
        return

    print(f" Building FAISS index from: {catalog_path}")
    build_index(catalog_path)
    print("FAISS index + product_ids.txt saved in /data")

if __name__ == "__main__":
    main()
