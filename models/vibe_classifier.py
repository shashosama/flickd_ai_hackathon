# vibe_classifier.py
from transformers import pipeline
import json

# Load vibes from your file
with open("data/vibeslist.json") as f:
    vibe_labels = json.load(f)

# Load zero-shot classifier (cached automatically)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_vibe(caption: str):
    result = classifier(caption, vibe_labels, multi_label=True)
    # Filter top N with confidence > 0.4
    vibes = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.4]
    return vibes[:3]
