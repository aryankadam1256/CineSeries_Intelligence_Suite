## CineSeries Intelligence Suite – Setup & Usage Commands

### 1. Navigate to project

```powershell
cd C:\Users\admin\Downloads\aryan\BACKEND\CineSeries_Intelligence_Suite
```

### 2. Create virtual environment (Windows, Python 3.13)

```powershell
C:\Users\admin\AppData\Local\Microsoft\WindowsApps\python3.13.exe -m venv .venv
```

### 3. Activate virtual environment

```powershell
.\.venv\Scripts\activate
```

To deactivate later:

```powershell
deactivate
```

### 4. Install project dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Install extra/dev tools (for tests and debugging)

```powershell
python -m pip install httpx pytest
```

### 6. Run the FastAPI server

From the project root with venv active:

```powershell
python -m uvicorn api.main:app --reload
```

Server URL:

- API docs (Swagger UI): `http://127.0.0.1:8000/docs`
- Simple UI (custom page): `http://127.0.0.1:8000/`

### 7. Run tests

```powershell
python tests/test_api.py
# or, if using pytest:
pytest -v
```

### 8. Useful git commands (remote + branch)

```powershell
git init
git remote add origin <YOUR_GITHUB_HTTPS_OR_SSH_URL>

# See all remotes and branches
git remote -v
git fetch origin
git branch -a

# Create local branch tracking remote review-sentiment-api
git checkout -b review-sentiment-api origin/review-sentiment-api

# Push local branch
git push -u origin review-sentiment-api

# Check current branch
git status
```

---

### 2. Simplify the UI: single textbox → sentiment

Let’s make `/` a very simple page: one textarea, one button, show “Sentiment: positive/negative” — no need to touch `/docs`.

Update `api/main.py` like this (only the important parts shown):

```python:api/main.py
# ... existing imports ...
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from api.utils import preprocess_text, get_sentiment_from_logits
from fastapi.responses import HTMLResponse  # <-- add this

app = FastAPI()

MODEL_PATH = "models/sentiment_model"
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
```

Now your flow is:

- Start server: `python -m uvicorn api.main:app --reload`
- Open `http://127.0.0.1:8000/`
- Type any review text → click **Analyze** → see `Sentiment: positive/negative` on the page.
