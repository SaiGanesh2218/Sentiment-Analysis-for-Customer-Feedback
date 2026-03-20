# ======================================================
# PROJECT: Sentiment Analysis for Customer Feedback (Enhanced + Fixed)
# ======================================================

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk import download
import warnings
warnings.filterwarnings("ignore")

# Download NLTK resources
download('vader_lexicon')
download('stopwords')

# ------------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# ------------------------------------------------------
np.random.seed(42)

def generate_synthetic_feedback(n=10000):
    positive_texts = [
        "I love this product!", "Excellent quality and fast delivery.",
        "Customer support was great!", "Highly satisfied with my purchase.",
        "Amazing experience, will buy again!"
    ]
    negative_texts = [
        "Terrible service.", "Product quality was poor.",
        "Not worth the money.", "Customer support was unhelpful.",
        "I want a refund immediately!"
    ]
    neutral_texts = [
        "The product is okay.", "Received the item as described.",
        "Average experience.", "Delivery took time but okay.",
        "Product matches the description."
    ]

    sentiments = []
    feedbacks = []
    for _ in range(n):
        sentiment_choice = np.random.choice(["positive", "negative", "neutral"], p=[0.45, 0.35, 0.20])
        if sentiment_choice == "positive":
            text = np.random.choice(positive_texts)
        elif sentiment_choice == "negative":
            text = np.random.choice(negative_texts)
        else:
            text = np.random.choice(neutral_texts)
        feedbacks.append(text)
        sentiments.append(sentiment_choice)

    dates = pd.date_range("2023-01-01", periods=n, freq="H")
    df = pd.DataFrame({
        "feedback_id": range(1, n+1),
        "feedback_text": feedbacks,
        "sentiment": sentiments,
        "date": np.random.choice(dates, n)
    })
    return df

df = generate_synthetic_feedback(n=10000)
print("✅ Dataset Generated — Shape:", df.shape)
print(df['sentiment'].value_counts(normalize=True))

# ------------------------------------------------------
# STEP 2: Text Preprocessing
# ------------------------------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\d+", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text.strip()

df["clean_text"] = df["feedback_text"].apply(clean_text)

# ------------------------------------------------------
# STEP 3: Add Extra NLP Features
# ------------------------------------------------------
sia = SentimentIntensityAnalyzer()

def add_extra_features(df):
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["char_count"] = df["clean_text"].apply(len)
    df["excl_count"] = df["feedback_text"].apply(lambda x: x.count("!"))
    df["contains_refund"] = df["clean_text"].apply(lambda x: int("refund" in x))
    df["contains_delay"] = df["clean_text"].apply(lambda x: int("delay" in x))
    df["vader_score"] = df["clean_text"].apply(lambda x: sia.polarity_scores(x)["compound"])
    return df

df = add_extra_features(df)

# ------------------------------------------------------
# STEP 4: Feature Extraction (TF-IDF + Numeric Features)
# ------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(df["clean_text"]).toarray()

# Combine text features + numeric ones
numeric_features = [
    "word_count", "char_count", "excl_count",
    "contains_refund", "contains_delay", "vader_score"
]
X_num = df[numeric_features].values

# ✅ FIX: Use MinMaxScaler to avoid negative values (for Naive Bayes compatibility)
scaler = MinMaxScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine both
X = np.hstack((X_text, X_num_scaled))
y = df["sentiment"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ------------------------------------------------------
# STEP 5: Train Ensemble Model (Logistic Regression + Naive Bayes)
# ------------------------------------------------------
lr = LogisticRegression(max_iter=500)
nb = MultinomialNB()
ensemble = VotingClassifier(estimators=[("lr", lr), ("nb", nb)], voting="soft")

print("\n🚀 Training Ensemble Model...")
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# ------------------------------------------------------
# STEP 6: Evaluation
# ------------------------------------------------------
print("\n📊 Model Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred), 4))

# Confusion Matrix Heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Sentiment Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=df, palette="viridis")
plt.title("Sentiment Distribution in Dataset")
plt.show()

# ------------------------------------------------------
# STEP 7: Save Dataset & Model Output
# ------------------------------------------------------
df.to_excel("customer_feedback.xlsx", index=False)
df.to_csv("customer_feedback.csv", index=False)
print("\n💾 Saved dataset as 'customer_feedback.xlsx' and 'customer_feedback.csv'")

# ------------------------------------------------------
# STEP 8: Interactive Prediction Function
# ------------------------------------------------------
def predict_feedback(text):
    text_clean = clean_text(text)
    features = {
        "word_count": len(text_clean.split()),
        "char_count": len(text_clean),
        "excl_count": text.count("!"),
        "contains_refund": int("refund" in text_clean),
        "contains_delay": int("delay" in text_clean),
        "vader_score": sia.polarity_scores(text_clean)["compound"]
    }
    text_vec = vectorizer.transform([text_clean]).toarray()
    num_vec = scaler.transform([list(features.values())])
    final_vec = np.hstack((text_vec, num_vec))
    pred = ensemble.predict(final_vec)[0]
    print(f"\n🧠 Predicted Sentiment: {pred.upper()}")

# Example Predictions
predict_feedback("I am unhappy with the product and want a refund!")
predict_feedback("Awesome service and great quality!")
predict_feedback("The product is okay, nothing special.")
