from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Load dataset
train_df = pd.read_csv("train.csv").dropna(subset=["condition", "review", "rating"])

# Load DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to classify sentiment using DistilBERT
def analyze_sentiment_distilbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    positive_score = probabilities[0][1].item()
    
    return "positive" if positive_score > 0.6 else "negative"

@app.route("/")
def home():
    return jsonify({"message": "Welcome to Medicine Recommendation API!"})

@app.route("/recommend", methods=["GET"])
def recommend():
    condition = request.args.get("condition")
    if not condition:
        return jsonify({"error": "Please provide a condition."}), 400

    try:
        results = train_df[train_df['condition'].str.contains(condition, case=False, na=False)].copy()
        if results.empty:
            return jsonify({"message": "No medicines found for this condition."})

        results["sentiment_label"] = results["review"].apply(analyze_sentiment_distilbert)
        results = results.sort_values(by="rating", ascending=False).drop_duplicates(subset=['drugName'])
        recommendations = results[['drugName', 'rating', 'sentiment_label', 'review']].head(10).to_dict(orient="records")

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
