import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

def read_text_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    similarity = round(similarity, 2) * 100
    return similarity

# App layout
st.title("Text Similarity Calculator")
st.markdown("This app calculates the similarity between two texts.")

# Input boxes for user text
text1 = st.text_area("Paste Text 1 here:")
text2 = st.text_area("Paste Text 2 here:")

# Calculate similarity on button click
if st.button("Calculate Similarity"):
    if text1 and text2:
        similarity = calculate_similarity(text1, text2)
        st.write(f"The content similarity between the texts is: {similarity}%")
    else:
        st.warning("Please enter both texts to calculate similarity.")
