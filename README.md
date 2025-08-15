# 🛒 Amazon Reviews Sentiment Analyzer

A simple **AI-powered web app** that analyzes Amazon product reviews to determine whether they are **Positive**, **Neutral**, or **Negative** — along with a confidence score.  
Built using **Python, Natural Language Processing (NLP), and Machine Learning** with a **Naive Bayes Classifier**.

🚀 **[Live Demo](https://sentiment-analyzer-yc9ljmwoasdk55f53qg7ym.streamlit.app/)**

---

## ✨ Features
- **Real-time sentiment prediction** from user input.
- **Confidence score visualization** for better understanding.
- **Interactive dashboard** built with Streamlit.
- Option to **track analysis history**.

---

## 🛠️ Tech Stack
- **Python** 🐍
- **NLTK** (Text preprocessing)
- **Scikit-learn** (Model training)
- **Pandas & NumPy** (Data handling)
- **Streamlit** (Web UI)

---

## 📂 Project Structure
sentiment-analyzer/
├── streamlit_app.py # Main Streamlit app
├── predict.py # Prediction logic (optional)
├── requirements.txt # Dependencies
├── README.md # Project documentation
├── models/
│ └── naive_bayes_model.pkl # Trained ML model
└── vectorizer.pkl # Text vectorizer

---

## 💻 Run Locally
```bash
# 1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-analyzer.git
cd sentiment-analyzer

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Start the app
streamlit run streamlit_app.py
The app will be available at http://localhost:8501.

🖼 Example

Input: "This product exceeded my expectations!"
Output: Positive (Confidence: 92%)

📚 Learning Outcomes:
Applied text preprocessing techniques for sentiment analysis.
Trained and evaluated a Naive Bayes classifier.
Deployed a machine learning model using Streamlit.
