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
- Wrote the data visualisation function (`plot_distribution`) that gives the spam vs. ham bar chart.
- Did the README.

**Mohammed Janoudi**
- Implemented model evaluation (`evaluate_model`): accuracy score and a formatted confusion matrix printed to the terminal.
- Implemented message prediction with confidence scores (`predict_message`) using `predict_proba`.
- Built the interactive prediction loop (`interactive_loop`) with quit handling and empty-input validation.
- Put everything together in `main.py` and tested the full program.

## 3. Difficulties

| Difficulty | How We Overcame It |
|------------|--------------------|
| Computers can't read raw text, so we had to understand TF-IDF before any model could train. | Read the scikit-learn `TfidfVectorizer` docs and tutorials; understood that it weights words by how informative they are, then applied the same fitted vectorizer to both training data and new messages. |
| The confusion matrix rows/columns kept flipping because scikit-learn orders labels alphabetically (`ham`, `spam`). | Passed an explicit `labels=['spam', 'ham']` argument to `confusion_matrix` so the printed table always matches our Spam/Ham headers. |
| Getting a confidence score, not just a label. | Used `model.predict_proba()` and took the maximum probability of the predicted class, then converted it to a percentage. |

## 4. Time Breakdown

| Task | Estimated | Actual |
|------|-----------|--------|
| Understanding TF-IDF | 2h | 2h |
| Environment setup | 1h | 0.5h |
| Dataset loading | 1h | 0.5h |
| Model training | 2h | 1.75h |
| Model evaluation | 2h | 2h |
| Visualization | 1h | 1.5h |
| Testing and debugging | 2h | 1.5h |
| **Total** | **11h** | **9.75h** |
