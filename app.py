import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Load and clean dataset

def load_data():
    df = pd.read_csv("spam.csv", encoding="latin1")[['Category', 'Message']]
    df.columns = ['label', 'text']
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Train model and vectorizer
def train_model():
    df = load_data()
    df['clean_text'] = df['text'].apply(clean_text)
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label_num']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()  # You can replace this with LogisticRegression() or LinearSVC()
    model.fit(X_train, y_train)
    return model, tfidf

# Streamlit UI
st.title("📩 Email Spam Classifier")
st.markdown("Classify your message as **Spam** or **Ham**.")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        model, vectorizer = train_model()
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]
        if result == 1:
            st.error("🚫 Spam Detected!")
        else:
            st.success("✅ This is a Ham message (not spam).")
