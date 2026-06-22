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


# 1. Dataset loading

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[['label', 'message']].dropna()
    df['label'] = df['label'].str.strip().str.lower()
    return df


# 2. Data visualisation 

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


# 3. Feature extraction + model training

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


# 4. Model evaluation 

def evaluate_model(model, X_test_tfidf, y_test) -> None:
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.1f}%")

    # Build the confusion matrix with an explicit label order so the printed rows/columns always match the Spam / Ham headers below.
    labels = ['spam', 'ham']
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("              Spam    Ham")
    print(f"Actual Spam   {cm[0][0]:>4}   {cm[0][1]:>4}")
    print(f"Actual Ham    {cm[1][0]:>4}   {cm[1][1]:>4}")


# 5. Single-message prediction with confidence

def predict_message(message: str, model, vectorizer) -> tuple[str, float]:
    # Transform the raw text with the SAME vectorizer used during training.
    features = vectorizer.transform([message])

    label = model.predict(features)[0]

    # predict_proba gives a probability for each class; the confidence is the probability assigned to the predicted label.
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100

    return label.upper(), confidence


# 6. Interactive prediction loop 

def interactive_loop(model, vectorizer) -> None:
    print("\nType 'quit' to exit.")

    while True:
        message = input("\nEnter message:\n").strip()

        if message.lower() == 'quit':
            print("Goodbye!")
            break

        if not message:
            print("Please enter a non-empty message.")
            continue

        label, confidence = predict_message(message, model, vectorizer)
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.1f}%")

