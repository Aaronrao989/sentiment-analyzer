import streamlit as st
import pickle
import re
import string
import math
from collections import Counter
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import nltk
from nltk.stem import SnowballStemmer 
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize

# Page configuration
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .confidence-bar {
        background: linear-gradient(90deg, #667eea, #764ba2);
        height: 20px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize NLTK components (with error handling)
@st.cache_resource
def load_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return SnowballStemmer('english'), set(stopwords.words('english'))
    except:
        return None, set()

stemmer, stop_words = load_nltk_data()

# Load model function
@st.cache_resource
def load_model(model_path='models/naive_bayes_model.pkl'):
    """Load the trained sentiment analysis model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Using demo mode with simulated predictions.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Text preprocessing function
def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    if stemmer:
        words = word_tokenize(text)
        words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 2]
    else:
        words = [w for w in text.split() if len(w) > 2]
    
    return ' '.join(words)

# Prediction functions
def make_class_prediction(text, counts, class_prob, class_count):
    """Make prediction for a single class using log probabilities"""
    log_prediction = math.log(class_prob)
    text_counts = Counter(re.split(r"\s+", text))
    total_words = sum(counts.values()) + class_count
    
    for word in text_counts:
        word_prob = (counts.get(word, 0) + 1) / total_words
        log_prediction += text_counts[word] * math.log(word_prob)
    
    return log_prediction

def simulate_sentiment_analysis(text):
    """Simulate sentiment analysis for demo purposes"""
    positive_words = ['great', 'excellent', 'amazing', 'fantastic', 'wonderful', 'perfect', 
                     'love', 'best', 'awesome', 'outstanding', 'superb', 'brilliant', 'good', 
                     'nice', 'beautiful', 'happy', 'satisfied', 'recommend', 'impressed', 'pleased']
    negative_words = ['terrible', 'awful', 'bad', 'worst', 'horrible', 'disappointing', 
                     'poor', 'useless', 'waste', 'broken', 'defective', 'cheap', 'fraud', 
                     'scam', 'hate', 'angry', 'frustrated', 'regret', 'unhappy', 'dissatisfied']
    
    words = text.lower().split()
    positive_score = sum(1 for word in words if any(pw in word for pw in positive_words))
    negative_score = sum(1 for word in words if any(nw in word for nw in negative_words))
    
    # Add some randomness
    positive_score += len(words) * 0.1 + (hash(text) % 100) / 200
    negative_score += len(words) * 0.1 + (hash(text[::-1]) % 100) / 200
    
    total = positive_score + negative_score
    if total == 0:
        total = 1
    
    pos_confidence = positive_score / total
    neg_confidence = negative_score / total
    
    # Normalize to ensure they sum to 1
    total_conf = pos_confidence + neg_confidence
    pos_confidence /= total_conf
    neg_confidence /= total_conf
    
    return {
        'sentiment': 'POSITIVE' if positive_score > negative_score else 'NEGATIVE',
        'positive_confidence': pos_confidence,
        'negative_confidence': neg_confidence
    }

def predict_sentiment(text, model):
    """Predict sentiment using the trained model or simulation"""
    processed_text = preprocess_text(text)
    
    if model:
        # Real model prediction
        neg_pred = make_class_prediction(
            processed_text, 
            model['negative_counts'], 
            model['prob_negative'], 
            model['negative_review_count']
        )
        
        pos_pred = make_class_prediction(
            processed_text, 
            model['positive_counts'], 
            model['prob_positive'], 
            model['positive_review_count']
        )
        
        # Calculate confidence scores
        pos_confidence = math.exp(pos_pred - max(pos_pred, neg_pred))
        neg_confidence = math.exp(neg_pred - max(pos_pred, neg_pred))
        
        # Normalize
        total = pos_confidence + neg_confidence
        pos_confidence /= total
        neg_confidence /= total
        
        prediction = "POSITIVE" if pos_pred > neg_pred else "NEGATIVE"
        
        return {
            'sentiment': prediction,
            'positive_confidence': pos_confidence,
            'negative_confidence': neg_confidence
        }
    else:
        # Simulation mode
        return simulate_sentiment_analysis(text)

# Initialize session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# Load model
model = load_model()

# Main UI
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
        avg_confidence = sum(max(item['positive_confidence'], item['negative_confidence']) 
                           for item in st.session_state.analysis_history) / total_analyses
        
        st.metric("Total Analyzed", total_analyses)
        st.metric("Positive Reviews", positive_count)
        st.metric("Negative Reviews", negative_count)
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("No analyses yet. Try analyzing some reviews!")
    
    st.markdown("---")
    st.markdown("### 🔧 Model Information")
    if model:
        st.success("✅ Trained model loaded")
        st.info("Using Naive Bayes classifier")
    else:
        st.warning("⚠️ Demo mode active")
        st.info("Using simulated predictions")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Your Review")
    
    # Text input
    review_text = st.text_area(
        "Type or paste an Amazon product review here:",
        height=150,
        placeholder="Example: This product exceeded my expectations! The quality is amazing and shipping was super fast. Highly recommended!"
    )
    
    # Example reviews
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
    
    # Use example text if selected
    if 'example_text' in st.session_state:
        review_text = st.session_state.example_text
        del st.session_state.example_text
    
    # Analysis button
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
            result = predict_sentiment(review_text, model)
            
            # Store in history
            st.session_state.analysis_history.append({
                'text': review_text[:100] + "..." if len(review_text) > 100 else review_text,
                'sentiment': result['sentiment'],
                'positive_confidence': result['positive_confidence'],
                'negative_confidence': result['negative_confidence'],
                'timestamp': datetime.now()
            })
            
            # Display result
            sentiment = result['sentiment']
            pos_conf = result['positive_confidence']
            neg_conf = result['negative_confidence']
            
            # Sentiment badge
            if sentiment == 'POSITIVE':
                st.markdown(f'<p class="positive-sentiment">✅ {sentiment}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="negative-sentiment">❌ {sentiment}</p>', unsafe_allow_html=True)
            
            # Confidence metrics
            st.metric("Positive Confidence", f"{pos_conf:.1%}")
            st.metric("Negative Confidence", f"{neg_conf:.1%}")
            
            # Confidence visualization
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Positive', 'Negative'],
                y=[pos_conf * 100, neg_conf * 100],
                marker_color=['#28a745', '#dc3545'],
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

# Analysis History
if st.session_state.analysis_history:
    st.header("📈 Analysis History")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(st.session_state.analysis_history)
    df['max_confidence'] = df[['positive_confidence', 'negative_confidence']].max(axis=1)
    
    # Display recent analyses
    st.subheader("Recent Analyses")
    for i, item in enumerate(reversed(st.session_state.analysis_history[-5:])):
        with st.expander(f"{item['sentiment']} - {item['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Text:** {item['text']}")
            st.write(f"**Sentiment:** {item['sentiment']}")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Positive:** {item['positive_confidence']:.1%}")
            with col2:
                st.write(f"**Negative:** {item['negative_confidence']:.1%}")
    
    # Visualization of history
    if len(st.session_state.analysis_history) > 1:
        st.subheader("Sentiment Trends")
        
        # Pie chart of sentiments
        sentiment_counts = df['sentiment'].value_counts()
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={'POSITIVE': '#28a745', 'NEGATIVE': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence over time
        fig_line = px.line(
            df.reset_index(),
            x='index',
            y='max_confidence',
            title='Confidence Scores Over Time',
            labels={'index': 'Analysis Number', 'max_confidence': 'Max Confidence'}
        )
        st.plotly_chart(fig_line, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit | Amazon Reviews Sentiment Analysis</p>
    <p>Deploy this app to <a href="https://streamlit.io/cloud" target="_blank">Streamlit Cloud</a> for free!</p>
</div>
""", unsafe_allow_html=True)