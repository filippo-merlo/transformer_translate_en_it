#%% ### Byte-Pair Encoding tokenization ###
''' from https://huggingface.co/learn/nlp-course/en/chapter6/8?fw=pt#building-a-wordpiece-tokenizer-from-scratch '''

from config import *

# Load the dataset
from datasets import load_dataset
ds = load_dataset("yhavinga/ccmatrix", "en-it", cache_dir=CACHE_DIR)
dataset_iter = iter(ds['train'])

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

it_tokenizer.train_from_iterator(get_training_corpus(italian_sentences), trainer=trainer)
eng_tokenizer.train_from_iterator(get_training_corpus(english_sentences), trainer=trainer)

print(it_tokenizer.encode("Ciao, come stai?".lower()).tokens)
print(eng_tokenizer.encode("Hello, how are you?".lower()).tokens)