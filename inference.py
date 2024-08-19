#%%
import torch 
import numpy as np
from config import *
from model import *
from tqdm import tqdm


def predict(TOKENIZATION_LEVEL,english_sentences,italian_sentences, START_TOKEN,PADDING_TOKEN,END_TOKEN):
  import os 
  from tqdm import tqdm
  max_sequence_length = 200

  if TOKENIZATION_LEVEL == 'character':
    TOKENIZER_ENC = None
    TOKENIZER_DEC = None
    italian_vocabulary = [
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
    it_index_to_vocabulary = {k:v for k,v in enumerate(italian_vocabulary)}
    en_index_to_vocabulary = {k:v for k,v in enumerate(italian_vocabulary)}

    it_vocabulary_to_index = {v:k for k,v in enumerate(italian_vocabulary)}
    en_vocabulary_to_index = {v:k for k,v in enumerate(italian_vocabulary)}

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
        if is_valid_length(italian_sentence, max_sequence_length)\
        and is_valid_length(english_sentence, max_sequence_length)\
        and is_valid_tokens(italian_sentence, italian_vocabulary)\
        and is_valid_tokens(english_sentence, italian_vocabulary):
            valid_sentence_indicies.append(index)

    print(f"Number of sentences: {len(italian_sentences)}")
    print(f"Number of valid sentences: {len(valid_sentence_indicies)}")


    italian_sentences = [italian_sentences[i] for i in valid_sentence_indicies]
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]
        
  elif TOKENIZATION_LEVEL == 'word':
    import nltk 
    from nltk.tokenize import word_tokenize
    nltk.download('punkt_tab')

    def custom_tokenizer(sentence):
            sentence = sentence.replace(' ', "<SPACE>")
            sentence = word_tokenize(sentence)
            i = 0
            result = []
            while i < len(sentence):
                if sentence[i:i+3] == ['<', 'SPACE', '>']:
                    result.append(' ')
                    i += 3  # Skip the next two elements
                else:
                    result.append(sentence[i])
                    i += 1
            return result

    TOKENIZER_ENC = custom_tokenizer
    TOKENIZER_DEC = custom_tokenizer

    import json
    with open(os.path.join(CACHE_DIR, 'it_vocabulary_to_index_word.json'), 'r') as json_file:
        it_vocabulary_to_index = json.load(json_file)
        print('Italian Vocabulary to Index',len(it_vocabulary_to_index))

    with open(os.path.join(CACHE_DIR, 'en_vocabulary_to_index_word.json'), 'r') as json_file:
        en_vocabulary_to_index = json.load(json_file)
        print('English Vocabulary to Index',len(en_vocabulary_to_index))
    
    # set max sequence length and filter out sentences that are too long or have invalid tokens
    max_sequence_length = 200

    # The sentence is less than the max sequence length
    def is_valid_length(sentence, max_sequence_length):
        return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

    # Filter out sentences that are too long or have invalid tokens
    valid_sentence_indicies = []
    for index in tqdm(range(len(italian_sentences))):
        italian_sentence, english_sentence = italian_sentences[index], english_sentences[index]
        if is_valid_length(italian_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length):
            valid_sentence_indicies.append(index)

    print(f"Number of sentences: {len(italian_sentences)}")
    print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

    italian_sentences = [italian_sentences[i] for i in valid_sentence_indicies]
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

  elif TOKENIZATION_LEVEL == 'word_piece':

    from tokenizers import (
          Tokenizer
      )
    it_tokenizer = Tokenizer.from_file(os.path.join(CACHE_DIR,"it_tokenizer.json"))
    eng_tokenizer = Tokenizer.from_file(os.path.join(CACHE_DIR,"eng_tokenizer.json"))

    def custom_it_tokenizer(sentence):
        return it_tokenizer.encode(sentence).tokens
    
    def custom_eng_tokenizer(sentence):
        return eng_tokenizer.encode(sentence).tokens
    
    TOKENIZER_ENC = custom_eng_tokenizer
    TOKENIZER_DEC = custom_it_tokenizer

    # Get index to character and character to index mappings

    it_vocabulary_to_index = it_tokenizer.get_vocab()
    print('Italian Vocabulary to Index',len(it_vocabulary_to_index))
    en_vocabulary_to_index = eng_tokenizer.get_vocab()
    print('English Vocabulary to Index',len(en_vocabulary_to_index))

    italian_vocabulary = list(it_vocabulary_to_index.keys())
    english_vocabulary = list(en_vocabulary_to_index.keys())

    it_index_to_vocabulary = {k:v for k,v in it_vocabulary_to_index.items()}
    en_index_to_vocabulary = {k:v for k,v in en_vocabulary_to_index.items()}

    # set max sequence length and filter out sentences that are too long or have invalid tokens
    max_sequence_length = 200

    # The sentence is less than the max sequence length
    def is_valid_length(sentence, max_sequence_length):
        return len(list(sentence)) < (max_sequence_length - 1) # need to re-add the end token so leaving 1 space

    # Filter out sentences that are too long or have invalid tokens
    valid_sentence_indicies = []
    for index in tqdm(range(len(italian_sentences))):
        italian_sentence, english_sentence = italian_sentences[index], english_sentences[index]
        if is_valid_length(italian_sentence, max_sequence_length) \
        and is_valid_length(english_sentence, max_sequence_length):
            valid_sentence_indicies.append(index)

    print(f"Number of sentences: {len(italian_sentences)}")
    print(f"Number of valid sentences: {len(valid_sentence_indicies)}")

    italian_sentences = [italian_sentences[i] for i in valid_sentence_indicies]
    english_sentences = [english_sentences[i] for i in valid_sentence_indicies]

  device = torch.device("cuda")
  max_sequence_length = 200

  d_model = 512
  batch_size = 30
  ffn_hidden = 2048
  num_heads = 8
  drop_prob = 0.1
  num_layers = 1
  max_sequence_length = 200
  it_vocab_size = len(italian_vocabulary)

  transformer = Transformer(d_model, 
                            ffn_hidden,
                            num_heads, 
                            drop_prob, 
                            num_layers, 
                            max_sequence_length,
                            it_vocab_size,
                            en_vocabulary_to_index,
                            it_vocabulary_to_index,
                            START_TOKEN, 
                            END_TOKEN, 
                            PADDING_TOKEN,
                            TOKENIZER_ENC,
                            TOKENIZER_DEC).to(device)

  transformer.load_state_dict(torch.load(os.path.join(MODEL_PATH,f"transformer_model_{TOKENIZATION_LEVEL}_level_tok.pth")))
  transformer.eval()

  # A large negative constant used to represent negative infinity in mask calculations.
  NEG_INFTY = -1e9

  def create_masks(eng_batch, it_batch, tokenizer_enc = None, tokenizer_dec = None, max_sequence_length = 200):
      # Determine the number of sentences in the batch.
      num_sentences = len(eng_batch)
      
      # Create a look-ahead mask for the decoder to prevent it from attending to future tokens.
      look_ahead_mask = torch.full([max_sequence_length, max_sequence_length], True)  # Initialize with True (masking)
      look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)  # Mask the upper triangle (excluding diagonal)

      # Initialize padding masks for the encoder and decoder with False (no masking)
      encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
      decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)
      decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length], False)

      # Iterate over each sentence in the batch
      for idx in range(num_sentences):
          # Get the length of the English and Italian sentences
          if tokenizer_enc and tokenizer_dec:
              eng_sentence_length = len(tokenizer_enc(eng_batch[idx]))
              it_sentence_length = len(tokenizer_dec(eng_batch[idx]))
          else:
              eng_sentence_length, it_sentence_length = len(eng_batch[idx]), len(it_batch[idx])
          
          # Identify the positions in the sequence that should be masked (i.e., padding positions)
          eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
          it_chars_to_padding_mask = np.arange(it_sentence_length + 1, max_sequence_length)
          
          # Apply padding masks for the encoder (prevent attention to padding tokens)
          encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
          encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
          
          # Apply padding masks for the decoder's self-attention mechanism
          decoder_padding_mask_self_attention[idx, :, it_chars_to_padding_mask] = True
          decoder_padding_mask_self_attention[idx, it_chars_to_padding_mask, :] = True
          
          # Apply padding masks for the decoder's cross-attention mechanism (attention to encoder output)
          decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
          decoder_padding_mask_cross_attention[idx, it_chars_to_padding_mask, :] = True

      # Convert padding masks into attention masks using NEG_INFTY (large negative value)
      encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
      decoder_self_attention_mask = torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
      decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
      
      # Return the computed attention masks for the encoder and decoder
      return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

  import time
  def translate(eng_sentence, max_sequence_length = 200):
    eng_sentence = (eng_sentence,)
    it_sentence = ("", )
    it_ids = []
    time_score = []
    for word_counter in range(max_sequence_length):
      encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, it_sentence)
      start_time = time.time()
      predictions = transformer(eng_sentence,
                                it_sentence,
                                encoder_self_attention_mask.to(device), 
                                decoder_self_attention_mask.to(device), 
                                decoder_cross_attention_mask.to(device),
                                enc_start_token=False,
                                enc_end_token=False,
                                dec_start_token=True,
                                dec_end_token=False)
      end_time = time.time()
      elapsed_time = end_time - start_time
      time_score.append(elapsed_time)

      next_token_prob_distribution = predictions[0][word_counter]
      next_token_index = torch.argmax(next_token_prob_distribution).item()
      if TOKENIZATION_LEVEL == 'word_piece':
        if it_tokenizer.decode([next_token_index]) == END_TOKEN:
          break
        it_ids.append(next_token_index)
        it_sentence = (it_tokenizer.decode(it_ids), )
      else:
        next_token = it_index_to_vocabulary[next_token_index]
        if next_token == END_TOKEN:
          break
        it_sentence = (it_sentence[0] + next_token, )
      print(eng_sentence)
      print(it_sentence[0])
    return it_sentence[0], np.mean(time_score)
     
  # compute BLEU SCORE
  from nltk.translate.bleu_score import sentence_bleu
  from nltk.translate.bleu_score import SmoothingFunction
  smoothie = SmoothingFunction().method4

  def bleu_score(pred_sentences, it_sentences):
    blue_scores = []
    for pred_sentence, it_sentence in tqdm(zip(pred_sentences, it_sentences)):
      pred_sentence = pred_sentence.split()
      it_sentence = it_sentence.split()
      score = sentence_bleu([pred_sentence], it_sentence, smoothing_function=smoothie)
      blue_scores.append(score)
    return np.mean(blue_scores)


  target_sentences_l50 = []
  predicted_sentences_l50 = []
  target_sentences_l100 = []
  predicted_sentences_l100 = []
  target_sentences_l150 = []
  predicted_sentences_l150 = []
  target_sentences_l200 = []
  predicted_sentences_l200 = []
  
  time_score = []
  for i in tqdm(range(len(english_sentences))):
    english_sentence = english_sentences[i]
    if len(english_sentence) <= 50:
      target_sentences_l50.append(italian_sentences[i])
      pred_sentece, pred_avg_time = translate(english_sentence)
      time_score.append(pred_avg_time)
      predicted_sentences_l50.append(pred_sentece)
    elif len(english_sentence) <= 100 and len(english_sentence) > 50:
      target_sentences_l100.append(italian_sentences[i])
      pred_sentece, pred_avg_time = translate(english_sentence)
      time_score.append(pred_avg_time)
      predicted_sentences_l100.append(pred_sentece)
    elif len(english_sentence) <= 150 and len(english_sentence) > 100:
      target_sentences_l150.append(italian_sentences[i])
      pred_sentece, pred_avg_time = translate(english_sentence)
      time_score.append(pred_avg_time)
      predicted_sentences_l150.append(pred_sentece)
    elif len(english_sentence) > 150 and len(english_sentence) <= 200:
      target_sentences_l200.append(italian_sentences[i])
      pred_sentece, pred_avg_time = translate(english_sentence)
      time_score.append(pred_avg_time)
      predicted_sentences_l200.append(pred_sentece)
    
  score_mean_l50 = bleu_score(predicted_sentences_l50, target_sentences_l50)
  score_mean_l100 = bleu_score(predicted_sentences_l100, target_sentences_l100)
  score_mean_l150 = bleu_score(predicted_sentences_l150, target_sentences_l150)
  score_mean_l200 = bleu_score(predicted_sentences_l200, target_sentences_l200)

  mean_time = np.mean(time_score)

  return score_mean_l50, score_mean_l100, score_mean_l150, score_mean_l200, mean_time


#%% GET THE DATA
# Load the dataset
from datasets import load_dataset
ds = load_dataset("yhavinga/ccmatrix", "en-it", cache_dir=CACHE_DIR)
dataset_iter = iter(ds['train'])

# Define vocabulary
# 1 Character based tokenization
START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

TOTAL_SENTENCES = 203000
TRAINING_SENTENCES = 202900

italian_sentences = []
english_sentences = []
for i in tqdm(range(TOTAL_SENTENCES)):
      if i <= TRAINING_SENTENCES:
          continue
      else:
        example = next(dataset_iter)
        italian_sentences.append(example['translation']['it'].lower())
        english_sentences.append(example['translation']['en'].lower())
          
tokenization_levels = ['character','word_piece','word']

for tokenization_level in tokenization_levels:
  print(tokenization_level)
  score_mean_l50, score_mean_l100, score_mean_l150, score_mean_l200, mean_time = predict(tokenization_level,english_sentences,italian_sentences,START_TOKEN,PADDING_TOKEN,END_TOKEN)
  print(f"Tokenization Level: {tokenization_level}")
  print(f"Score Mean L50: {score_mean_l50}")
  print(f"Score Mean L100: {score_mean_l100}")
  print(f"Score Mean L150: {score_mean_l150}")
  print(f"Score Mean L200: {score_mean_l200}")
  print(f"Mean Time: {mean_time}")
  print("\n")