# 📩 Email-Spam-Classifier

A simple and interactive web application built using **Streamlit** to classify email/SMS messages as **Spam** or **Ham** (Not Spam).  
This project uses **Natural Language Processing (NLP)** and **Machine Learning (ML)** with **TFIDF** ,**Bag of Words** approach using **trigrams**.

---

## 🧠 Features

- Cleaned and preprocessed text data (lowercase, punctuation removal, stemming, stopword removal).
- Uses **CountVectorizer** with `ngram_range=(3,3)` (trigrams).
- Trains and compares 3 ML models:
  - **Multinomial Naive Bayes**
  - **Logistic Regression**
  - **Support Vector Machine (SVM)**
- Interactive prediction interface.
- Choose your preferred model in the app.
- Detects spam in real-time.

---

## 🗂️ Dataset

The dataset used is `spam.csv`, containing labeled SMS messages as either **ham** (not spam) or **spam**.

- 📄 Columns:
  - `Category`: ham or spam
  - `Message`: message content

---

## 🧪 Sample Output

- Input: `Congratulations! You've won a free cruise. Call now!`  
  Output: 🚫 **Spam Detected!**

- Input: `Hey, are we still meeting for lunch tomorrow?`  
  Output: ✅ **This is a Ham message (not spam).**

---

## 📦 Libraries Used

- `streamlit`
- `pandas`
- `nltk`
- `scikit-learn`

---

## 📁 File Structure

```
📦 spam-classifier/
├── spam.csv
├── app.py
└── README.md
```

---
