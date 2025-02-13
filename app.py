from flask import Flask, request, render_template, jsonify
from pyngrok import ngrok
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Initialize Flask app
app = Flask(__name__)


# Start Ngrok for public access
flask_tunnel = ngrok.connect(5000, "http")
flask_url = flask_tunnel.public_url
print(f"🚀 Flask API is Live at: {flask_url}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    condition = request.args.get("condition")

    if not condition:
        return jsonify({"error": "Please provide a condition."}), 400

    try:
        results = train_df[train_df['condition'].str.contains(condition, case=False, na=False)].copy()

        if results.empty:
            return jsonify({"message": "No medicines found for this condition."})

        # Analyze sentiment using DistilBERT
        results["sentiment_label"], results["sentiment_score"] = zip(*results["review"].apply(analyze_sentiment_distilbert))

        # Sort by sentiment score
        results = results.sort_values(by="sentiment_score", ascending=False)

        # Remove duplicate medicines
        results = results.drop_duplicates(subset=['drugName'])

        # Convert to JSON
        recommendations = results[['drugName', 'rating', 'sentiment_label', 'sentiment_score', 'usefulCount', 'review']].head(10).to_dict(orient="records")

        return jsonify(recommendations)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
