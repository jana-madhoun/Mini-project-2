# Mini-project-2 — Spam Detector

COMP 472 Mini Project 2. Classifies email/SMS messages as SPAM or HAM using TF-IDF + Logistic Regression.

## Dependencies

```bash
pip install pandas scikit-learn matplotlib seaborn
```

## Usage

Run interactively (prompts you to enter messages one at a time):

```bash
python3 main.py
```

## How It Works

1. Loads and cleans the dataset (`spam_clean.csv`)
2. Extracts features using TF-IDF vectorization (top 5000 terms, sublinear TF scaling)
3. Trains a Logistic Regression classifier (80/20 train/test split)
4. Evaluates accuracy and prints a confusion matrix to the terminal
5. Enters an interactive loop where you can type any message and get a SPAM/HAM prediction with confidence %

## Team

| Name | Student ID |
|------|------------|
| Jana El Madhoun | 40272201 |
| Mohammed Janoudi | 40252494 |