import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

# Load and process data
data = pd.read_csv("D:\Programs\python\ml project\spam.csv")
data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])

# Split into training & testing sets
mess = data['Message']
cat = data['Category']
(mess_train, mess_test, cat_train, cat_test) = train_test_split(mess, cat, test_size=0.2)

# Text vectorization
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Train model
model = MultinomialNB()
model.fit(features, cat_train)

# Define prediction function
def predict(message):
    input_message = cv.transform([message]).toarray()  # Fix: Use the actual input
    result = model.predict(input_message)
    return result[0]  # Fix: Extract single string value

# Streamlit UI
st.header('Spam Detection')
input_mess = st.text_input('Enter Message Here')

if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(f"### Prediction: {output}")  # Fix: Extract first value
