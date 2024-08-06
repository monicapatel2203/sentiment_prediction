import tensorflow as tf
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, load_tokenizer
import json

from preprocess import preprocess_tweet

app = FastAPI()

class SentimentReq(BaseModel):
	tweet: str

MODEL_WEIGHTS_PATH = 'model2.weights.h5'
TOKENIZER_DIR = "vocabulary"
VOCAB_SIZE = 20000
MAX_LEN = 64

model = load_model(MODEL_WEIGHTS_PATH, MAX_LEN, VOCAB_SIZE)
vectorizer = load_tokenizer(TOKENIZER_DIR, MAX_LEN, VOCAB_SIZE)

inp = np.random.randint(0, 10, (2, 64))
oup = model(inp)
print(model.summary())


def get_prediction(tweet):
	tokens = vectorizer(tweet)
	tokens = tf.expand_dims(tokens, axis=0)

	pred = model.predict(tokens, verbose=0)
	label = np.argmax(pred, axis=-1)
	return int(label[0])

@app.get("/")
def greet():
	return {
		"status": "Sentiment Prediction"
	}

@app.post("/predict_sentiment")
def predict_sentiment(req: SentimentReq):
	tweet = req.tweet

	# Preprocess the tweet as done while training the model
	tweet = preprocess_tweet(tweet)

	# Get the prediction from the model
	result = get_prediction(tweet)

	return {
		"response": result
	}