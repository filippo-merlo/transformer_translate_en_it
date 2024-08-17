#%% ### Byte-Pair Encoding tokenization ###
''' from https://huggingface.co/learn/nlp-course/en/chapter6/8?fw=pt#building-a-wordpiece-tokenizer-from-scratch '''

from config import *

# Load the dataset
from datasets import load_dataset
ds = load_dataset("yhavinga/ccmatrix", "en-it", cache_dir=CACHE_DIR)
dataset_iter = iter(ds['train'])

# Get subset of sentences from dataset: 200000
from tqdm import tqdm 
TOTAL_SENTENCES = 23000
scores = []
english_sentences = []
italian_sentences = []
for _ in tqdm(range(TOTAL_SENTENCES)):
    example = next(dataset_iter)
    italian_sentences.append(example['translation']['it'].lower())
    english_sentences.append(example['translation']['en'].lower())

def get_training_corpus(sentences):
    for i in range(0, len(sentences), 1000):
        yield ' '.join(sentences[i : i + 1000])

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

it_tokenizer = Tokenizer(models.WordPiece(unk_token="<UNK>"))
eng_tokenizer = Tokenizer(models.WordPiece(unk_token="<UNK>"))


it_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase()]
)
eng_tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)

it_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)
eng_tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

special_tokens = ["<UNK>", '<PADDING>', '<START>', '<END>']
trainer = trainers.WordPieceTrainer(vocab_size=10000, special_tokens=special_tokens)

it_tokenizer.train_from_iterator(get_training_corpus(italian_sentences), trainer=trainer)
eng_tokenizer.train_from_iterator(get_training_corpus(english_sentences), trainer=trainer)

it_tokenizer.decoder = decoders.WordPiece(prefix="##")
eng_tokenizer.decoder = decoders.WordPiece(prefix="##")

print('Sentence')
print("Gli angeli e la pappà che è così fresca.")
it_encoding = it_tokenizer.encode("Gli angeli e la pappà che è così fresca.".lower())
print('Sentence enc')
print(it_encoding.tokens)
print('ids')
print(type(it_encoding.ids))
print(it_encoding.ids)
it_decoding = it_tokenizer.decode(it_encoding.ids)
print('Sentence dec')
print(it_decoding)

import os
it_tokenizer.save(os.path.join(CACHE_DIR,"it_tokenizer.json"))
eng_tokenizer.save(os.path.join(CACHE_DIR,"eng_tokenizer.json"))
