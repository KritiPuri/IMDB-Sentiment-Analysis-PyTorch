# IMDB Simple RNN Sentiment Analysis 

## Model
- IMDB dataset: 25,000 training and 25,000 test reviews
- Vocabulary: up to 10,000 tokens, built only from training reviews
- Sequence length: 500
- Embedding: 128 dimensions
- Simple RNN: 128 hidden units, ReLU
- Fully connected output: 1 logit
- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Validation split: 20% of training data
- Early stopping: patience of 3 epochs

## Install (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Train
```bash
python train.py
```
The IMDB archive is downloaded and extracted automatically on the first run. Best validation model is saved to `simple_rnn_imdb.pth`.

Options:
```bash
python train.py --epochs 5 --batch-size 64 --max-len 500
```

## Predict
```bash
python prediction.py
```

## Streamlit interface
```bash
streamlit run app.py
```

## Project files
- `data_utils.py`: download/extract IMDB, tokenize, build vocabulary, encode and pad
- `train.py`: PyTorch RNN, training, validation, early stopping, testing
- `sentiment.py`: preprocessing and prediction utilities
- `prediction.py`: terminal prediction demo
- `app.py`: Streamlit web UI
