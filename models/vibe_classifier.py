from transformers import pipeline
#importing the pipeline from HUgging face transformers 

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#load the zero-shot classification pipeline using the BART large MNLI model
#Define a list of possible fashion "vibes" or aesthetics to classify into 
vibe_labels = [
    "Coquette", "Clean Girl", "Grunge", "Boho", "Cottagecore",
    "Streetwear", "Y2K", "Minimalist", "Vintage", "Edgy",
    "Chic", "Preppy", "Soft Girl", "Baddie", "E-Girl"
]
#define a function to classify the vibe of given text caption 
def classify_vibe(caption: str):
    #if the caption is empty or only whitespace , return a default label 
    if not caption or caption.strip() == "":
        print("[WARN] Empty caption received. Returning default vibe.")
        return ["Unspecified"]
    #use the classifier to predict which vibe labels match the caption 
    #multi_label = true allows multiple vibes to predicted 
    result = classifier(caption, vibe_labels, multi_label=True)
    #filter out vibe labels with scores above a threshold 
    vibes = [
        label for label, score in zip(result["labels"], result["scores"]) 
        if score > 0.4
    ]
    #if any vibes are confidently matched , return them ; otherwise return a fallback 
    return vibes if vibes else ["Unspecified"]
