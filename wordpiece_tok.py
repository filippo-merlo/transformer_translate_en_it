#%% ### Byte-Pair Encoding tokenization ###
''' from https://huggingface.co/learn/nlp-course/en/chapter6/8?fw=pt#building-a-wordpiece-tokenizer-from-scratch '''

# Import necessary libraries
from dataset import *


english_corpus = ['' + ' ' + s for s in english_sentences][0]
italian_corpus = ['' + ' ' + s for s in italian_sentences][0]

def get_training_corpus(corpus):
    for i in range(0, len(dataset), 1000):
        yield corpus[i : i + 1000]

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

it_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
eng_tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))


eng_tokenizer.normalizer = normalizers.Sequence(
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

special_tokens = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

it_tokenizer.train_from_iterator(get_training_corpus(italian_corpus), trainer=trainer)
eng_tokenizer.train_from_iterator(get_training_corpus(english_corpus), trainer=trainer)

print(it_tokenizer.encode("Ciao, come stai?").tokens)
print(eng_tokenizer.encode("Hello, how are you?").tokens)