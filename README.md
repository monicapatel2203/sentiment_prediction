# sentiment_prediction

Here we made a sentiment classifier that can identify the attitude of a given tweet by using Transformer. Here I 
have used a non-causal transformer that can look the entire sentence all at once during its processing. 

In addition to that I have used pretrained Glove word embedding of 100-Dimension for vectorizing a word and later
used that word representation along with transformer to make prediction. 

You can check the working of the app here: [Sentiment Predictor](https://senti-predictor.streamlit.app/)

The backend is deployed on hugging face: [Backend](https://huggingface.co/spaces/monica22/sentiment_prediction?logs=container)

### Live Demo
[sentiment_predictor.webm](https://github.com/user-attachments/assets/c116d28b-d660-486a-bf02-be0f69af7dc4)
