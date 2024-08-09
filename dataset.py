#%%
# Import Libraries
from config import *

# Define vocabulary
# 1 Character based tokenization
START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

german_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
    'y', 'z', 
    'ä', 'ö', 'ü', 'ß',  # German-specific characters
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
]

dutch_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
    'y', 'z',
    'à', 'ä', 'ë', 'ï', 'ö', 'ü', 'é', 'è', 'ê', 'ç', 
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
]

english_vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    ':', '<', '=', '>', '?', '@',
    '[', '\\', ']', '^', '_', '`', 
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
    'y', 'z', 
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
]

vocabulary = [
    START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '–','—',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
    ':', '<', '=', '>', '?', '@', ';', 
    '[', '\\', ']', '^', '_', '`', '‘', '’', '“', '”', '…', '«', '»',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'l', 
    'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'z', 
    'j', 'k', 'w', 'x', 'y', 
    'à', 'é', 'è', 'ì', 'ò', 'ù',
    '{', '|', '}', '~', PADDING_TOKEN, END_TOKEN
]

# Get index to character and character to index mappings
index_to_vocabulary = {k:v for k,v in enumerate(vocabulary)}
vocabulary_to_index = {v:k for k,v in enumerate(vocabulary)}

#%%
# Load the dataset
from datasets import load_dataset
ds = load_dataset("yhavinga/ccmatrix", "en-it", cache_dir=CACHE_DIR)
dataset_iter = iter(ds['train'])

#%%
# Get subset of sentences from dataset: 200000
from tqdm import tqdm 
TOTAL_SENTENCES = 200000
scores = []
english_sentences = []
italian_sentences = []
for _ in tqdm(range(TOTAL_SENTENCES)):
    example = next(dataset_iter)
    italian_sentences.append(example['translation']['it'].lower())
    english_sentences.append(example['translation']['en'].lower())
    scores.append(example['score'])

print(len(english_sentences))
print(len(italian_sentences))

#%%
# Limit Number of sentences
import numpy as np
# set the max sentence length based on the 99th percentile
PERCENTILE = 99
print( f"{PERCENTILE}th percentile length Italian: {np.percentile([len(x) for x in italian_sentences], PERCENTILE)}" )
print( f"{PERCENTILE}th percentile length English: {np.percentile([len(x) for x in english_sentences], PERCENTILE)}" )

# 99th percentile length Italian: 179.0
# 99th percentile length English: 166.0

#%%
# set max sequence length and filter out sentences that are too long or have invalid tokens
max_sequence_length = 200

# All tokens in the sentence are in the vocabulary
def is_valid_tokens(sentence, vocab):
    for token in list(set(sentence)):
        if token not in vocab:
            return False
    return True

# The sentence is less than the max sequence length
def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

# Filter out sentences that are too long or have invalid tokens
valid_sentence_indicies = []
for index in range(len(italian_sentences)):
    italian_sentence, english_sentence = italian_sentences[index], english_sentences[index]
    if is_valid_length(italian_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(italian_sentence, vocabulary)\
      and is_valid_tokens(english_sentence, vocabulary):
        valid_sentence_indicies.append(index)

print(f"Number of sentences: {len(italian_sentences)}")
print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


italian_sentences = [italian_sentences[i] for i in valid_sentence_indicies]
english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

#%%
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):

    def __init__(self, english_sentences, italian_sentences):
        self.english_sentences = english_sentences
        self.italian_sentences = italian_sentences

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        return self.english_sentences[idx], self.italian_sentences[idx]
    

dataset = TextDataset(english_sentences, italian_sentences)
