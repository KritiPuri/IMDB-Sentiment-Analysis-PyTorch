import torch
from train import SentimentRNN
from data_utils import tokenize, pad_sequences

def load_model(model_path="simple_rnn_imdb.pth"):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = SentimentRNN(
        checkpoint["vocab_size"],
        checkpoint["embed_dim"],
        checkpoint["hidden_size"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["vocab"], checkpoint["max_len"]

def preprocess_text(text, vocab, max_len):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    padded = pad_sequences([ids], max_len)
    return torch.tensor(padded, dtype=torch.long)

def predict_sentiment(review, model, vocab, max_len):
    inputs = preprocess_text(review, vocab, max_len)
    with torch.no_grad():
        probability = torch.sigmoid(model(inputs)).item()
    label = "Positive" if probability >= 0.5 else "Negative"
    return label, probability
