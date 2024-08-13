#%% ### Byte-Pair Encoding tokenization ###
''' from https://huggingface.co/learn/nlp-course/en/chapter6/5#training-algorithm '''

# Import necessary libraries
from dataset import *

english_corpus = english_sentence
italian_corpus = italian_sentence

print(english_corpus[:10])
# Import gpt2 tokenizer:
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

#%% compute the frequencies of each word in the corpus as we do the pre-tokenization:
from collections import defaultdict

word_freqs = defaultdict(int)

for text in english_corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)