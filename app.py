import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Amazon Review Sentiment Analyzer",
    page_icon="🛒",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styling */
    .info-card {
        background-color: #01182e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Result cards */
    .result-positive {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.3);
    }
    
    .result-negative {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 4px 12px rgba(235, 51, 73, 0.3);
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .confidence-badge {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
        font-size: 1.1rem;
        backdrop-filter: blur(10px);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Example reviews styling */
    .example-reviews {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .example-title {
        font-weight: 600;
        color: #495057;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">🛒 Amazon Review Sentiment Analyzer</div>
        <div class="header-subtitle">Powered by Machine Learning • Instant Analysis</div>
    </div>
""", unsafe_allow_html=True)

# Info card
st.markdown("""
    <div class="info-card">
        <strong>📊 How it works:</strong><br>
        Enter any product review below and our AI model will analyze the sentiment 
        to determine if it's positive or negative, along with a confidence score.
    </div>
""", unsafe_allow_html=True)

# ----------------------------
# Load model & vectorizer
# ----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("sentiment_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# NLTK setup
# ----------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words("english"))

# ----------------------------
# Text preprocessing
# ----------------------------
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# ----------------------------
# UI Input
# ----------------------------
st.markdown("### ✍️ Enter Your Review")
review = st.text_area(
    "Type or paste an Amazon product review:",
    height=150,
    placeholder="Example: This product exceeded my expectations! The quality is outstanding and delivery was fast. Highly recommend to anyone looking for...",
    label_visibility="collapsed"
)

# Example reviews section
with st.expander("💡 Need inspiration? Try these examples"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Positive Examples:**")
        st.markdown("- *Amazing product! Worth every penny.*")
        st.markdown("- *Best purchase I've made this year.*")
        st.markdown("- *Excellent quality and fast shipping.*")
    with col2:
        st.markdown("**Negative Examples:**")
        st.markdown("- *Terrible quality, broke after one use.*")
        st.markdown("- *Waste of money, very disappointed.*")
        st.markdown("- *Poor customer service and defective item.*")

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner("🔄 Analyzing sentiment..."):
            processed = preprocess_text(review)
            vectorized = vectorizer.transform([processed])
            prediction = model.predict(vectorized)[0]
            probability = model.predict_proba(vectorized)[0][1]
            
            if prediction == 1:
                st.markdown(f"""
                    <div class="result-positive">
                        <div class="result-title">✅ Positive Review</div>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                            This review expresses positive sentiment!
                        </p>
                        <div class="confidence-badge">
                            Confidence: {probability:.1%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-negative">
                        <div class="result-title">❌ Negative Review</div>
                        <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                            This review expresses negative sentiment.
                        </p>
                        <div class="confidence-badge">
                            Confidence: {(1-probability):.1%}
                        </div>
                    </div>
                """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
    <div class="footer">
        <strong>🤖 Model:</strong> Naive Bayes Classifier | 
        <strong>🛠️ Built with:</strong> Streamlit & scikit-learn<br>
        Made with ❤️ for sentiment analysis
    </div>
""", unsafe_allow_html=True)