# Review Sentiment API - Project Explanation

## Overview

This microservice implements sentiment analysis for movie reviews using state-of-the-art transformer models (e.g., BERT/RoBERTa). It acts as an API endpoint where users can submit movie review text and receive a sentiment classification in response.

---

## Architecture Summary

- Client sends POST requests with review text.
- API receives text, preprocesses it, and feeds it into a pre-trained transformer fine-tuned for sentiment classification.
- The model outputs sentiment predictions (positive, negative, neutral).
- API sends back the prediction in a structured JSON format.

This design uses Flask or FastAPI as the service framework and loads a transformer model saved after offline fine-tuning.

---

## File Structure and Roles

- `api/main.py`: Main API app entry - defines routes, loads model, handles incoming requests, calls preprocessing + prediction, returns response.
- `api/utils.py`: Contains helper functions - text preprocessing (cleaning, tokenization), token-to-model input conversion, model inference functions.
- `models/sentiment_model.bin`: Serialized transformer model file saved after training.
- `notebooks/sentiment_training.ipynb`: Jupyter notebook for data loading, preprocessing, fine-tuning the transformer on movie review datasets, and evaluation.
- `data/movie_reviews.csv`: Sample dataset (e.g., IMDB reviews) used for training and validation.
- `tests/test_api.py`: Unit and integration tests for API correctness.
- `requirements.txt`: Python dependencies needed for the service.

---

## Flow from Request to Response

1. A user sends a movie review text to the API's `/predict` endpoint.
2. `main.py` receives this POST request and extracts text.
3. Text is passed to `utils.py` for preprocessing - tokenization and formatting for transformer input.
4. Preprocessed input is forwarded to the loaded transformer model for inference.
5. The model returns sentiment predictions.
6. `main.py` formats this prediction into a JSON response.
7. Response sent back to the user.

---

## Dataset Selection

For the first iteration, the **IMDB Movie Reviews Dataset** is recommended due to its balanced, benchmark sentiment classification labels and simplicity. Alternative or supplemental datasets include the Stanford Sentiment Treebank for more granular sentiment nuances.

---

## Next Steps

- Implement API skeleton with Flask/FastAPI.
- Prepare and explore IMDB dataset.
- Fine-tune transformer model using provided notebook template.
- Integrate the fine-tuned model back into the API for inference.
- Write tests for API robustness.
- Deploy to free cloud platforms like Railway or Hugging Face Spaces.

---

This explanation provides a clear and concise understanding of the Review Sentiment API project module. Save it as a reference to accompany your codebase and project documentation.
