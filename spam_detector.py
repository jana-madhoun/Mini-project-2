"""
Spam Detection AI — COMP 472 Mini Project 2
Classifies email/SMS messages as SPAM or HAM using TF-IDF + Logistic Regression.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ── 1. Dataset loading ────────────────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[['label', 'message']].dropna()
    df['label'] = df['label'].str.strip().str.lower()
    return df


# ── 2. Data visualisation ─────────────────────────────────────────────────────

def plot_distribution(df: pd.DataFrame, save_path: str = 'distribution.png') -> None:
    counts = df['label'].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(counts.index.str.upper(), counts.values, color=['green', 'red'])
    plt.title('Spam vs Ham')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ── 3. Feature extraction + model training ────────────────────────────────────

def train_model(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, sublinear_tf=True)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(C=100, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer, X_test_tfidf, y_test


# ── 4. Model evaluation ───────────────────────────────────────────────────────

def evaluate_model(model, X_test_tfidf, y_test) -> None:
    # TODO: predict on X_test_tfidf, print accuracy and confusion matrix to terminal
    raise NotImplementedError("evaluate_model not implemented")


# ── 5. Single-message prediction with confidence ──────────────────────────────

def predict_message(message: str, model, vectorizer) -> tuple[str, float]:
    # TODO: vectorize message, predict label and confidence, return (LABEL, confidence_percent)
    raise NotImplementedError("predict_message not implemented")


# ── 6. Interactive prediction loop ────────────────────────────────────────────

def interactive_loop(model, vectorizer) -> None:
    # TODO: loop reading user input, call predict_message, print label + confidence; exit on 'quit'
    raise NotImplementedError("interactive_loop not implemented")

