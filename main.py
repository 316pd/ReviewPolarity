import streamlit as st
import pickle
import re
from bs4 import BeautifulSoup
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def clean_text(reviews):
    reviews = re.sub(r"http\S+", "", reviews)
    reviews = BeautifulSoup(reviews, 'lxml').get_text()
    reviews = re.sub(r"won't", "will not", reviews)
    reviews = re.sub(r"can\'t", "can not", reviews)
    reviews = re.sub(r"n\'t", " not", reviews)
    reviews = re.sub(r"\'re", " are", reviews)
    reviews = re.sub(r"\'s", " is", reviews)
    reviews = re.sub(r"\'d", " would", reviews)
    reviews = re.sub(r"\'ll", " will", reviews)
    reviews = re.sub(r"\'t", " not", reviews)
    reviews = re.sub(r"\'ve", " have", reviews)
    reviews = re.sub(r"\'m", " am", reviews)
    reviews = re.sub("\S*\d\S*", "", reviews).strip()
    reviews = re.sub('[^A-Za-z]+', ' ', reviews)
    reviews = ' '.join(e.lower() for e in reviews.split() if e.lower() not in stopwords)
    return [reviews.strip()]

loaded_model = pickle.load(open('model.sav', 'rb'))
loaded_vectorizer = pickle.load(open('tfidf.sav','rb'))

st.title('Polarity of Review')
review = st.text_input("Reviews:", value="")
if st.button('Predict'):
    if loaded_model.predict(loaded_vectorizer.transform(clean_text(review)))[0] == 0:
        st.write('Given Review is Negative')
    else:
        st.write('Given Review is Positive')