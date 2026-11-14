import streamlit as st
import pickle

# Load the saved model and vectorizer
with open("fake_news_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Website title
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

st.title("📰 Fake News Detection System")

st.write("Enter a news headline or article below to check its predicted category.")

# Input box
news_text = st.text_area("Paste your news article here:")

# Prediction button
if st.button("Check News"):

    if news_text.strip() == "":
        st.warning("Please enter some news text first.")

    else:
        # Convert text into numerical features
        transformed_text = vectorizer.transform([news_text])

        # Predict the category
        prediction = model.predict(transformed_text)[0]

        if prediction == 0:
            st.error("Prediction: LIKELY FAKE NEWS")

        else:
            st.success("Prediction: LIKELY REAL NEWS")

# Disclaimer
st.info(
    "Note: This system uses machine learning patterns "
    "from a labelled dataset. It does not independently "
    "verify the factual accuracy of news."
)
