import argparse
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data_utils import download_and_extract, read_split, build_vocab, encode, pad_sequences

MAX_LEN = 500
EMBED_DIM = 128
HIDDEN_SIZE = 128
VOCAB_SIZE = 10000

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True, nonlinearity="relu")
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden[-1]).squeeze(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--output", default="simple_rnn_imdb.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root = download_and_extract()
    train_texts, train_labels = read_split(root, "train")
    test_texts, test_labels = read_split(root, "test")

    # Use only the training split to create the vocabulary.
    vocab = build_vocab(train_texts, VOCAB_SIZE)
    train_ids = pad_sequences([encode(text, vocab) for text in train_texts], args.max_len)
    test_ids = pad_sequences([encode(text, vocab) for text in test_texts], args.max_len)

    train_x = torch.tensor(train_ids, dtype=torch.long)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    test_x = torch.tensor(test_ids, dtype=torch.long)
    test_y = torch.tensor(test_labels, dtype=torch.float32)

    # Create a validation split from training data.
    indices = torch.randperm(len(train_x))
    split = int(len(indices) * 0.8)
    train_idx, val_idx = indices[:split], indices[split:]
    train_ds = TensorDataset(train_x[train_idx], train_y[train_idx])
    val_ds = TensorDataset(train_x[val_idx], train_y[val_idx])
    test_ds = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = SentimentRNN(len(vocab)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_val_loss, patience, wait = float("inf"), 3, 0

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
            total += labels.size(0)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                val_correct += ((torch.sigmoid(logits) >= 0.5).float() == labels).sum().item()
                val_total += labels.size(0)

        train_loss = running_loss / total
        val_loss /= val_total
        print(f"Epoch {epoch+1}/{args.epochs} | Train loss {train_loss:.4f} | Train acc {correct/total:.4f} | Val loss {val_loss:.4f} | Val acc {val_correct/val_total:.4f}")

        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab": vocab,
                "vocab_size": len(vocab),
                "embed_dim": EMBED_DIM,
                "hidden_size": HIDDEN_SIZE,
                "max_len": args.max_len
            }, args.output)
            print("Saved best model:", args.output)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probabilities = torch.sigmoid(model(inputs))
            test_correct += ((probabilities >= 0.5).float() == labels).sum().item()
            test_total += labels.size(0)
    print(f"Test accuracy: {test_correct/test_total:.4f}")

if __name__ == "__main__":
    main()
