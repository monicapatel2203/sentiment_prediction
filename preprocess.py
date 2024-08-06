import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lematizing = WordNetLemmatizer()

def preprocess_tweet(tweet):
	# Lowercase the tweet
    tweet = tweet.lower()
    
    # remove URLs
    tweet = re.sub(r'http\S+|www\S+', '', tweet)
    
    # remove Mention(@Username) from data
    tweet = re.sub(r'@\w+', '', tweet)
    
    # remove punctuation, special characters & emojis
    tweet = re.sub(r'[^\w\s]|[\u263a-\U0001f645]', '', tweet)
    
    # Remove any non-ASCII character
    tweet = re.sub(r'[^\x00-\x7F]+', '', tweet)
    
    tokens = word_tokenize(tweet)
    # remove stopwords like "the", "is", etc
    tokens = [x for x in tokens if x not in stop_words]
    
    tokens = [lematizing.lemmatize(x) for x in tokens]
    
    tweet = ' '.join(tokens)
    
    return tweet