from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

vibe_labels = [
    "Coquette", "Clean Girl", "Grunge", "Boho", "Cottagecore",
    "Streetwear", "Y2K", "Minimalist", "Vintage", "Edgy",
    "Chic", "Preppy", "Soft Girl", "Baddie", "E-Girl"
]

def classify_vibe(caption: str):
    if not caption or caption.strip() == "":
        print("[WARN] Empty caption received. Returning default vibe.")
        return ["Unspecified"]
    
    result = classifier(caption, vibe_labels, multi_label=True)
    vibes = [
        label for label, score in zip(result["labels"], result["scores"]) 
        if score > 0.4
    ]
    return vibes if vibes else ["Unspecified"]
