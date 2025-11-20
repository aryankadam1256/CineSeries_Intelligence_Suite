# api/utils.py

import torch

def preprocess_text(text, tokenizer):
    # Tokenize input text with padding and truncation
    encoded_input = tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    return encoded_input

def get_sentiment_from_logits(logits):
    # Convert logits to sentiment label
    predicted_class = torch.argmax(logits, dim=1).item()
    label_map = {0: "negative", 1: "positive"}
    return label_map.get(predicted_class, "unknown")
