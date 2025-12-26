import streamlit as st
import pickle
import re
import string
from nltk.stem import SnowballStemmer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
.main { padding-top: 2rem; }

.header-container {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 15px;
    margin-bottom: 2rem;
    color: white;
}

.result-positive {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.result-negative {
    background: linear-gradient(135deg, #eb3349, #f45c43);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.confidence {
    font-size: 1.1rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("""
<div class="header-container">
    <h1>🛒 Amazon Review Sentiment Analyzer</h1>
    <p>Machine Learning • Real-time Analysis</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Enter an Amazon product review and the model will predict "
    "whether the sentiment is **Positive** or **Negative**."
)

# ----------------------------
# Load model & vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Text preprocessing (SAFE)
# ----------------------------
stemmer = SnowballStemmer("english")

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()

    tokens = [
        stemmer.stem(word)
        for word in tokens
        if len(word) > 2
    ]

    return " ".join(tokens)

# ----------------------------
# User Input
# ----------------------------
review = st.text_area(
    "✍️ Enter your review:",
    height=150,
    placeholder="This product is amazing and totally worth the price!"
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        processed = preprocess_text(review)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0][1]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-positive">
                <h2>✅ Positive Review</h2>
                <div class="confidence">Confidence: {probability:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-negative">
                <h2>❌ Negative Review</h2>
                <div class="confidence">Confidence: {(1 - probability):.1%}</div>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption(
    "🤖 Model: Naive Bayes | 🛠 Built with Streamlit & scikit-learn"
)
