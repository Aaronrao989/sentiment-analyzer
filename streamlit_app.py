import os
import math
import re
import string
import pickle
from collections import Counter
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# ────────────────────────────────────────────────────────────────────────────────
# Page config + styles
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-bottom: 2rem;
        border-radius: 10px;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .positive-sentiment { color: #28a745; font-weight: bold; font-size: 1.2em; }
    .negative-sentiment { color: #dc3545; font-weight: bold; font-size: 1.2em; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────────
# NLTK setup (used for the legacy/custom model)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        # some environments require this; harmless if it doesn’t exist
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
        return SnowballStemmer('english'), set(stopwords.words('english'))
    except Exception:
        return None, set()

stemmer, stop_words = load_nltk_data()


# ────────────────────────────────────────────────────────────────────────────────
# Model loader (supports sklearn tuple or legacy dict)
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """
    Tries to load `sentiment_model.pkl` from the same folder as this script.
    Supports:
      1) (vectorizer, clf) scikit-learn pipeline parts
      2) legacy dict with keys: negative_counts, prob_negative, negative_review_count,
                                positive_counts, prob_positive, positive_review_count
    Returns a dict bundle or None.
    """
    model_path = os.path.join(os.path.dirname(__file__), "sentiment_model.pkl")
    try:
        with open(model_path, "rb") as f:
            obj = pickle.load(f)

        # sklearn case: tuple(vectorizer, model)
        if isinstance(obj, tuple) and len(obj) == 2:
            vectorizer, clf = obj
            return {"type": "sklearn", "vectorizer": vectorizer, "model": clf}

        # legacy/custom case: dict with counts
        required = {
            "negative_counts", "prob_negative", "negative_review_count",
            "positive_counts", "prob_positive", "positive_review_count"
        }
        if isinstance(obj, dict) and required.issubset(set(obj.keys())):
            obj["type"] = "custom"
            return obj

        st.warning("Model file loaded but format not recognized. Running in demo mode.")
        return None

    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Using demo mode with simulated predictions.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────────
# Preprocessing (used by the legacy/custom model)
# ────────────────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    if stemmer:
        words = word_tokenize(text)
        words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    else:
        words = [w for w in text.split() if len(w) > 2]
    return " ".join(words)


# ────────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ────────────────────────────────────────────────────────────────────────────────
def make_class_prediction(text, counts, class_prob, class_count):
    """Naive Bayes log-probability for legacy/custom model."""
    log_prediction = math.log(class_prob)
    text_counts = Counter(re.split(r"\s+", text))
    total_words = sum(counts.values()) + class_count
    for word in text_counts:
        word_prob = (counts.get(word, 0) + 1) / total_words
        log_prediction += text_counts[word] * math.log(word_prob)
    return log_prediction


def simulate_sentiment_analysis(text):
    """Simple fallback if model is missing."""
    positive_words = {'great','excellent','amazing','fantastic','wonderful','perfect',
                      'love','best','awesome','outstanding','superb','brilliant','good',
                      'nice','beautiful','happy','satisfied','recommend','impressed','pleased'}
    negative_words = {'terrible','awful','bad','worst','horrible','disappointing',
                      'poor','useless','waste','broken','defective','cheap','fraud',
                      'scam','hate','angry','frustrated','regret','unhappy','dissatisfied'}

    words = text.lower().split()
    p = sum(any(pw in w for pw in positive_words) for w in words) + 0.1 * len(words)
    n = sum(any(nw in w for nw in negative_words) for w in words) + 0.1 * len(words)

    total = max(p + n, 1e-9)
    pos_conf, neg_conf = p / total, n / total
    sentiment = "POSITIVE" if p >= n else "NEGATIVE"
    return {"sentiment": sentiment, "positive_confidence": pos_conf, "negative_confidence": neg_conf}


def predict_sentiment(text, bundle):
    """
    Predict using:
      - sklearn: vectorizer.transform + clf.(predict_proba|predict)
      - custom:   legacy NB with word counts
      - demo:     simulator
    """
    if bundle is None:
        return simulate_sentiment_analysis(text)

    if bundle["type"] == "sklearn":
        vec = bundle["vectorizer"]
        clf = bundle["model"]
        X = vec.transform([text])
        # prediction label
        pred = clf.predict(X)[0]
        sentiment = str(pred).upper()

        # try to compute confidences
        pos_conf = neg_conf = 0.5
        try:
            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X)[0]
                classes = list(clf.classes_)
                if "positive" in classes:
                    pos_idx = classes.index("positive")
                    pos_conf = float(probs[pos_idx])
                    if "negative" in classes:
                        neg_idx = classes.index("negative")
                        neg_conf = float(probs[neg_idx])
                    else:
                        neg_conf = 1.0 - pos_conf
                else:
                    # if labels are 1/-1 or similar
                    max_idx = int(probs.argmax())
                    pos_conf = float(probs[max_idx])
                    neg_conf = 1.0 - pos_conf
        except Exception:
            pass

        return {
            "sentiment": sentiment,
            "positive_confidence": pos_conf,
            "negative_confidence": neg_conf
        }

    # legacy/custom model
    processed = preprocess_text(text)
    neg_pred = make_class_prediction(
        processed, bundle['negative_counts'], bundle['prob_negative'], bundle['negative_review_count']
    )
    pos_pred = make_class_prediction(
        processed, bundle['positive_counts'], bundle['prob_positive'], bundle['positive_review_count']
    )

    pos_conf = math.exp(pos_pred - max(pos_pred, neg_pred))
    neg_conf = math.exp(neg_pred - max(pos_pred, neg_pred))
    total = pos_conf + neg_conf
    pos_conf /= total
    neg_conf /= total

    return {
        "sentiment": "POSITIVE" if pos_pred > neg_pred else "NEGATIVE",
        "positive_confidence": pos_conf,
        "negative_confidence": neg_conf
    }


# ────────────────────────────────────────────────────────────────────────────────
# Session state & model
# ────────────────────────────────────────────────────────────────────────────────
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

model_bundle = load_model()


# ────────────────────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 Amazon Reviews Sentiment Analyzer</h1>
    <p>Advanced AI-powered sentiment analysis for product reviews</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Analysis Statistics")

    total_analyses = len(st.session_state.analysis_history)
    if total_analyses > 0:
        positive_count = sum(1 for item in st.session_state.analysis_history if item['sentiment'] == 'POSITIVE')
        negative_count = total_analyses - positive_count
        avg_conf = sum(max(i['positive_confidence'], i['negative_confidence'])
                       for i in st.session_state.analysis_history) / total_analyses

        st.metric("Total Analyzed", total_analyses)
        st.metric("Positive Reviews", positive_count)
        st.metric("Negative Reviews", negative_count)
        st.metric("Avg Confidence", f"{avg_conf:.1%}")

        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analyses yet. Try analyzing some reviews!")

    st.markdown("---")
    st.markdown("### 🔧 Model Information")
    if model_bundle:
        if model_bundle["type"] == "sklearn":
            st.success("✅ Trained model loaded (scikit-learn)")
        else:
            st.success("✅ Trained model loaded (custom NB)")
    else:
        st.warning("⚠️ Demo mode active — using simulated predictions.")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Your Review")

    review_text = st.text_area(
        "Type or paste an Amazon product review here:",
        height=150,
        placeholder="Example: This product exceeded my expectations! The quality is amazing and shipping was super fast. Highly recommended!"
    )

    with st.expander("💡 Try These Examples"):
        examples = {
            "Positive Example 1": "Amazing product! The quality exceeded my expectations. Fast shipping and excellent customer service. Would definitely buy again and recommend to friends!",
            "Negative Example 1": "Terrible quality for the price. Product broke after just two days of use. Customer service was unhelpful and rude. Complete waste of money.",
            "Positive Example 2": "Love this purchase! Works exactly as described and the design is beautiful. Great value for money. The packaging was also very nice.",
            "Negative Example 2": "Not what I expected at all. Poor build quality and doesn't match the description. Returning this immediately. Very disappointed."
        }
        for label, text in examples.items():
            if st.button(label, key=label):
                st.session_state.example_text = text
                st.rerun()

    if 'example_text' in st.session_state:
        review_text = st.session_state.example_text
        del st.session_state.example_text

    col_analyze, col_clear = st.columns([1, 1])
    with col_analyze:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()

with col2:
    st.header("🎯 Analysis Result")

    if analyze_button and review_text.strip():
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(review_text, model_bundle)

            st.session_state.analysis_history.append({
                'text': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                'sentiment': result['sentiment'],
                'positive_confidence': result['positive_confidence'],
                'negative_confidence': result['negative_confidence'],
                'timestamp': datetime.now()
            })

            sentiment = result['sentiment']
            pos_conf = result['positive_confidence']
            neg_conf = result['negative_confidence']

            if sentiment == 'POSITIVE':
                st.markdown(f'<p class="positive-sentiment">✅ {sentiment}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="negative-sentiment">❌ {sentiment}</p>', unsafe_allow_html=True)

            st.metric("Positive Confidence", f"{pos_conf:.1%}")
            st.metric("Negative Confidence", f"{neg_conf:.1%}")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Positive', 'Negative'],
                y=[pos_conf * 100, neg_conf * 100],
                text=[f'{pos_conf:.1%}', f'{neg_conf:.1%}'],
                textposition='auto',
            ))
            fig.update_layout(
                title="Confidence Scores",
                yaxis_title="Confidence (%)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analyze_button and not review_text.strip():
        st.error("Please enter a review to analyze!")

# History + charts
if st.session_state.analysis_history:
    st.header("📈 Analysis History")

    df = pd.DataFrame(st.session_state.analysis_history)
    df['max_confidence'] = df[['positive_confidence', 'negative_confidence']].max(axis=1)

    st.subheader("Recent Analyses")
    for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"{item['sentiment']} - {item['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Text:** {item['text']}")
            st.write(f"**Sentiment:** {item['sentiment']}")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**Positive:** {item['positive_confidence']:.1%}")
            with c2:
                st.write(f"**Negative:** {item['negative_confidence']:.1%}")

    if len(st.session_state.analysis_history) > 1:
        st.subheader("Sentiment Trends")

        sentiment_counts = df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_line = px.line(
            df.reset_index(),
            x='index',
            y='max_confidence',
            title='Confidence Scores Over Time',
            labels={'index': 'Analysis Number', 'max_confidence': 'Max Confidence'}
        )
        st.plotly_chart(fig_line, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit | Amazon Reviews Sentiment Analysis</p>
    <p>Deploy this app to <a href="https://streamlit.io/cloud" target="_blank">Streamlit Cloud</a> for free!</p>
</div>
""", unsafe_allow_html=True)