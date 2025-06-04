from transformers import pipeline # importing libraries 

# all the  vibes 
VIBES = ["Coquette", "Clean Girl", "Cottagecore", "Streetcore", "Y2K", "Boho", "Party Glam"]

# Load zero-shot classifier pipeline
#the model did not see training data for these vibes but can read the caption and say : cottage and boho 
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

#function that reads and looks at the captoon or transcript from a video
def classify_vibe(text):
    result = classifier(text, candidate_labels=VIBES, multi_label=True)#multi_label = true so it can pick more than one if it wants 
    # Return top 3 vibes with score > 0.3
    vibes = [label for label, score in zip(result["labels"], result["scores"]) if score > 0.3]
    return vibes[:3]#sends back top 3 matching vibes for this video 
