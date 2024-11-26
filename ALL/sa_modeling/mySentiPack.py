import re
import unicodedata
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') #stopwords 

import spacy


# Function to clean tweets
def clean_tweet(df, col_tweet):
    # Convert to lowercase
    df[col_tweet] = df[col_tweet].str.lower()
    
    # Remove links from the text
    df[col_tweet] = df[col_tweet].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x, flags=re.MULTILINE))
  
    # Remove hashtags from the text
    df[col_tweet] = df[col_tweet].apply(lambda x: re.sub(r'\B#\S+', '', x))
  
    # Remove mentions (@user) from the text
    df[col_tweet] = df[col_tweet].apply(lambda x: re.sub(r'@\w+', '', x))
   
    # Remove accents from characters and keep only letters and spaces
    df[col_tweet] = df[col_tweet].apply(lambda x: ''.join(
        c for c in unicodedata.normalize('NFD', x) 
        if unicodedata.category(c) != 'Mn' and c.isalpha() or c.isspace()))
    
    # Substitute multiple spaces with a single space
    df[col_tweet] = df[col_tweet].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    
    


# Function to remove Spanish stopwords
def remove_stopwords(df, col_tweet):
    # Load Spanish stopwords
    stop_words = set(stopwords.words('spanish'))
    
    # Remove stopwords from the tweets
    df[col_tweet] = df[col_tweet].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in stop_words]))


# Function to lemmatize Spanish text
def lemmatize_text(df, col_tweet):
    # Load the Spanish language model 'es_core_news_sm' from spacy 
    nlp = spacy.load('es_core_news_sm')
    
    # Lemmatize the text in the specified column
    df[col_tweet] = df[col_tweet].apply(lambda x: ' '.join(
        [token.lemma_ for token in nlp(x)]))
        
        
# Function to update the removal of bad words
def remove_badwords(df, col_tweet, BadWordsList):
    # Load Spanish stopwords
    #stop_words = BadWordsList
    
    # Remove stopwords from the tweets
    df[col_tweet] = df[col_tweet].apply(lambda x: ' '.join(
        [word for word in x.split() if word not in BadWordsList]))
