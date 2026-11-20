from sentiment import load_model, predict_sentiment

model, vocab, max_len = load_model()
review = "This movie was fantastic! The acting was great and the plot was thrilling."
sentiment, probability = predict_sentiment(review, model, vocab, max_len)
print("Review:", review)
print("Sentiment:", sentiment)
print(f"Positive probability: {probability:.2%}")
