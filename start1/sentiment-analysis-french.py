import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
results = classifier("Nous sommes très heureux de vous présenter la bibliothèque 🤗 Transformers.")
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'], 4)}")