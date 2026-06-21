from spam_detector import load_dataset, plot_distribution, train_model, evaluate_model, interactive_loop


def main():
    print("Welcome to Spam Detection AI")
    print("Training model...")

    try:
        df = load_dataset('spam.csv')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found.")
        return

    plot_distribution(df)

    model, vectorizer, X_test_tfidf, y_test = train_model(df)

    evaluate_model(model, X_test_tfidf, y_test)

    interactive_loop(model, vectorizer)


if __name__ == '__main__':
    main()
