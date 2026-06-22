# Reflection — COMP 472 Mini Project 2

## 1. Team Members

| Name | Student ID |
|------|------------|
| Jana El Madhoun     |   40272201         |
| Mohammed Janoudi     |     40252494       |

## 2. Contributions

**Jana El Madhoun**
- Set up the project environment and installed the required libraries (pandas, numpy, scikit-learn, matplotlib, seaborn).
- Implemented dataset loading with pandas (`load_dataset`), including cleaning and normalising the label column.
- Built the feature extraction and model training pipeline (`train_model`): TF-IDF vectorisation, train/test split, and the Logistic Regression classifier.
- Wrote the data visualisation function (`plot_distribution`) that produces the spam vs. ham bar chart.
- Drafted the README.

**Mohammed Janoudi**
- Implemented model evaluation (`evaluate_model`): accuracy score and a formatted confusion matrix printed to the terminal.
- Implemented single-message prediction with confidence scores (`predict_message`) using `predict_proba`.
- Built the interactive prediction loop (`interactive_loop`) with quit handling and empty-input validation.
- Wired everything together in `main.py` and tested the full program end to end.

## 3. Difficulties

| Difficulty | How We Overcame It |
|------------|--------------------|
| Computers can't read raw text, so we had to understand TF-IDF before any model could train. | Read the scikit-learn `TfidfVectorizer` docs and tutorials; understood that it weights words by how informative they are, then applied the same fitted vectorizer to both training data and new messages. |
| The confusion matrix rows/columns kept flipping because scikit-learn orders labels alphabetically (`ham`, `spam`). | Passed an explicit `labels=['spam', 'ham']` argument to `confusion_matrix` so the printed table always matches our Spam/Ham headers. |
| Getting a confidence score, not just a label. | Used `model.predict_proba()` and took the maximum probability of the predicted class, then converted it to a percentage. |

## 4. Time Breakdown

| Component | Understanding | Development | Testing | Total |
|-----------|--------------|-------------|---------|-------|
| Dataset loading (`load_dataset`) | 0.5h | 0.5h | 0.5h | 1.5h |
| Data visualisation (`plot_distribution`) | 0.5h | 0.5h | 0.5h | 1.5h |
| Model training (`train_model`) | 2h | 1.5h | 0.5h | 4h |
| Model evaluation (`evaluate_model`) | 1h | 1h | 0.5h | 2.5h |
| Single-message prediction (`predict_message`) | 1h | 0.5h | 0.5h | 2h |
| Interactive loop (`interactive_loop`) | 0.5h | 0.5h | 0.5h | 1.5h |
| **Total** | **5.5h** | **4.5h** | **3h** | **13h** |
