import streamlit as st
import requests
import json

st.title("Sentiment Classification")

tweet = st.text_input("tweet")

CLASS_LABEL = {
	0: 'Negative', 
	1: 'Neutral', 
	2: 'Positive'
}

def predict():
	URL = "https://monica22-sentiment-prediction.hf.space//predict_sentiment"

	resp = requests.post(
			URL,
			json={"tweet": tweet}
		)

	result = json.loads(resp._content)
	sentiment = CLASS_LABEL[result['response']]

	st.write(f"Predicted sentiment is {sentiment}")

st.button("Predict", on_click=predict, type="primary")