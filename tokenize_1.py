
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import pickle
import numpy as np

input_length = 128

dataset = pd.read_csv('FinalCombinedDataset.csv')
dataset.dropna(inplace=True)
dataset.drop_duplicates(inplace=True)
X_data, y_data = np.array(dataset['texts']), np.array(dataset['category'])

vocab_length = 40000

tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_data)
tokenizer.num_words = vocab_length
print("Tokenizer vocab length:", vocab_length)

# Saving the tokenizer
import json

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
