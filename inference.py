import torch 
import numpy as np
from dataset import *
from config import *
from model import *
import os 

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

def create_masks(eng_batch, it_batch, tokenizer_enc = None, tokenizer_dec = None):
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

def translate(eng_sentence):
  eng_sentence = (eng_sentence,)
  it_sentence = ("", )
  it_ids = []
  for word_counter in range(max_sequence_length):
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, it_sentence)
    predictions = transformer(eng_sentence,
                              it_sentence,
                              encoder_self_attention_mask.to(device), 
                              decoder_self_attention_mask.to(device), 
                              decoder_cross_attention_mask.to(device),
                              enc_start_token=False,
                              enc_end_token=False,
                              dec_start_token=True,
                              dec_end_token=False)
    
    next_token_prob_distribution = predictions[0][word_counter]
    next_token_index = torch.argmax(next_token_prob_distribution).item()
    if TOKENIZATION_LEVEL == 'word_piece':
      if it_tokenizer.decode([next_token_index]) == END_TOKEN:
        print(it_tokenizer.decode([next_token_index]))
        print(END_TOKEN)
        break
      it_ids.append(next_token_index)
      it_sentence = (it_tokenizer.decode(it_ids), )
    else:
      next_token = it_index_to_vocabulary[next_token_index]
      if next_token == END_TOKEN:
        break
      it_sentence = (it_sentence[0] + next_token, )
  return it_sentence[0]
     
# compute BLUE SCORE
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
smoothie = SmoothingFunction().method4

def blue_score(pred_sentences, it_sentences):
  blue_scores = []
  for pred_sentence, it_sentence in tqdm(zip(pred_sentences, it_sentences)):
    pred_sentence = pred_sentence.split()
    it_sentence = it_sentence.split()
    score = sentence_bleu([pred_sentence], it_sentence, smoothing_function=smoothie)
    print(pred_sentence)
    print(it_sentence)
    print(score)
    blue_scores.append(score)
  return np.mean(blue_scores)


target_sentences = []
predicted_sentences = []

for i in tqdm(range(100)):
  english_sentence = english_sentences[i]
  target_sentences.append(italian_sentences[i])
  predicted_sentences.append(translate(english_sentences[i]))#.replace('<END>', ''))
score_mean = blue_score(predicted_sentences, target_sentences)
print(score_mean)