import re
import tarfile
import urllib.request
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data")
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
VOCAB_SIZE = 10000
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

def download_and_extract():
    DATA_DIR.mkdir(exist_ok=True)
    if not (DATA_DIR / "aclImdb").exists():
        if not ARCHIVE.exists():
            print("Downloading IMDB dataset...")
            urllib.request.urlretrieve(DATASET_URL, ARCHIVE)
        print("Extracting dataset...")
        with tarfile.open(ARCHIVE, "r:gz") as tar:
            tar.extractall(DATA_DIR, filter="data")
    return DATA_DIR / "aclImdb"

def tokenize(text):
    return re.findall(r"[a-z]+|[0-9]+|[^\w\s]", text.lower())

def read_split(root, split):
    texts, labels = [], []
    for label_name, label in (("neg", 0), ("pos", 1)):
        folder = root / split / label_name
        for path in sorted(folder.glob("*.txt")):
            texts.append(path.read_text(encoding="utf-8", errors="ignore"))
            labels.append(label)
    return texts, labels

def build_vocab(texts, max_features=VOCAB_SIZE):
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, _ in counts.most_common(max_features - len(vocab)):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]

def pad_sequences(sequences, max_len=500):
    result = []
    for sequence in sequences:
        sequence = sequence[-max_len:]
        result.append([0] * (max_len - len(sequence)) + sequence)
    return result
