from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from api.utils import preprocess_text, get_sentiment_from_logits
from fastapi.responses import HTMLResponse  # <-- add this

app = FastAPI()

MODEL_PATH = "models/sentiment_model-BERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# Simple page: enter review text and get sentiment
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Review Sentiment</title>
        </head>
        <body>
            <h2>Movie / Series Review Sentiment</h2>
            <textarea id="text" rows="5" cols="60"
                placeholder="Type your review here..."></textarea><br/>
            <button onclick="analyze()">Analyze</button>
            <p id="result"></p>

            <script>
            async function analyze() {
                const text = document.getElementById('text').value;
                const res = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text })
                });
                if (!res.ok) {
                    document.getElementById('result').textContent =
                        'Error: ' + res.status + ' ' + (await res.text());
                    return;
                }
                const data = await res.json();
                document.getElementById('result').textContent =
                    'Sentiment: ' + data.sentiment;
            }
            </script>
        </body>
    </html>
    """

class ReviewRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_sentiment(review: ReviewRequest):
    if not review.text or not review.text.strip():
        raise HTTPException(status_code=400, detail="Text input cannot be empty.")
    inputs = preprocess_text(review.text, tokenizer)
    with torch.no_grad():
        outputs = model(**inputs)
    sentiment = get_sentiment_from_logits(outputs.logits)
    return {"sentiment": sentiment}
# ... existing code ...