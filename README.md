


# Flickd AI Hackathon - Fashion Video Intelligence Engine

## Overview

This project was developed for the Flickd AI Hackathon. It focuses on analyzing fashion videos to detect clothing items, match them to products, and classify the style or “vibe” of the outfit. It combines computer vision and natural language processing to create a smart fashion insight tool.

## Problem Statement

In the world of online fashion and short videos, it’s difficult to identify and retrieve exact outfit items seen in influencer clips or fashion reels. This creates friction for users and limits opportunities for retail platforms.

### Key Challenges:

- Automatically detecting clothing and accessories from video frames.
- Matching these items to existing products despite style variations and lighting changes.
- Inferring an overall style or fashion vibe (e.g., “Chic”, “Streetwear”) from video content.
- Delivering all insights in a structured and usable format.

## Solution

We built a Python-based AI pipeline that:

1. **Extracts video frames** using OpenCV.
2. **Detects fashion items** using YOLOv8.
3. **Generates embeddings** using CLIP for each detected item.
4. **Matches items** to a pre-indexed fashion product database using FAISS.
5. **Classifies fashion vibe** using a zero-shot text classifier.
6. **Exports structured results** in `.json` format with labeled preview frames.

The project runs via a Python backend (Gradio or Streamlit UI) and is deployable to Hugging Face Spaces.

## Directory Structure

```
flikd-ai-hackathon/
├── app.py                 # Frontend (Gradio or Streamlit)
├── main.py                # Core pipeline
├── utils/
│   ├── frame_utils.py     # Frame extraction
│   ├── detection.py       # YOLOv8 wrapper
│   ├── clip_matcher.py    # CLIP + FAISS logic
│   └── vibe_classifier.py # Style classification
├── outputs/               # Output images + JSON
├── product_db/            # Product embeddings
├── requirements.txt
└── README.Rmd
```

## How to Use

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/flikd-ai-hackathon.git
cd flikd-ai-hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py         # for Gradio

# or
streamlit run app.py  # for Streamlit
```

- Upload a `.mp4` or `.mov` video file.
- The system will:
    - Extract key frames.
    - Detect clothing items.
    - Match to catalog.
    - Classify the vibe.
    - Return results in JSON.

## Sample Output

```json
{
  "video_id": "2025-05-27_13-46-16_UTC",
  "vibes": ["Chic"],
  "products": [
    {
      "type": "skirt",
      "match_type": "no match",
      "matched_product_id": "17124",
      "confidence": 0.686
    }
  ]
}
```

## What I Learned

This project helped me explore:

- **Computer Vision** using YOLOv8 and OpenCV.
- **Multimodal Embeddings** with CLIP for image-text alignment.
- **Semantic Search** with FAISS for scalable product matching.
- **Zero-Shot NLP** to classify fashion vibes from visual context.
- **Pipeline Deployment** using Python, Gradio/Streamlit, and Hugging Face Spaces.

It also improved my understanding of combining vision and language models in an end-to-end application for the fashion domain.

## Future Directions

- Integrate APIs from online retailers (e.g., Shopify, Zara).
- Fine-tune vibe classifiers with labeled datasets.
- Batch process multiple videos.
- Add feedback-based learning and personalization.
