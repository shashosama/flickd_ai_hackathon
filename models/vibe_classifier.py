from transformers import pipeline

# Define supported vibes
VIBES = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]

# Load zero-shot classifier pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def classify_vibe(text):
    result = classifier(text, candidate_labels=VIBES, multi_label=True)
    # Return top 3 vibes with score > 0.3
    vibes = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
    return vibes[:3]
