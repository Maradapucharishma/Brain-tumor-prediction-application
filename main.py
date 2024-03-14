import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

st.title("Emotion Detection using ML")
text = st.text_input("Enter the text")

model = pickle.load(open(r"C:\Users\CHARISHMA\Machine Learning\estimator.pkl",'rb'))
st.button("submit")
result = model.predict([text])




if result == 'Anger':
    st.image('https://freepngimg.com/thumb/angry_emoji/36886-1-angry-emoji-photo-thumb.png')
elif result == 'Love':
    st.image('https://freepngimg.com/thumb/emoji/64989-emoticon-heart-love-emoji-png-free-photo-thumb.png')
elif result == 'Joy':
    st.image('https://static.vecteezy.com/system/resources/thumbnails/029/138/681/small/happy-emoji-happy-emoji-happy-emoji-transparent-background-ai-generative-free-png.png')
elif result == 'Sad':
    st.image('https://freepngimg.com/thumb/sad_emoji/36860-2-sad-emoji-transparent-image-thumb.png')
elif result == 'Suprise':
    st.image('https://static.vecteezy.com/system/resources/thumbnails/009/665/371/small/emoticon-shocked-face-png.png')
elif result == 'Fear':
    st.image('https://static.vecteezy.com/system/resources/thumbnails/009/885/115/small/fearful-face-emoji-3d-illustration-png.png')