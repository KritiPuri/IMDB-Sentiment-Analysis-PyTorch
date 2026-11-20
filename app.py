import streamlit as st
from sentiment import load_model, predict_sentiment

st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬")
st.title("IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review and classify it as positive or negative.")

@st.cache_resource
def get_model():
    return load_model()

review = st.text_area("Movie Review", placeholder="Write your review here...", height=180)
if st.button("Classify Review", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        model, vocab, max_len = get_model()
        sentiment, probability = predict_sentiment(review, model, vocab, max_len)
        st.subheader(f"Sentiment: {sentiment}")
        st.metric("Positive probability", f"{probability:.2%}")
        st.progress(min(max(probability, 0.0), 1.0))
