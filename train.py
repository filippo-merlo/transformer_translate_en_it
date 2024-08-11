#%%
from model import Transformer
import torch 
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import *
from config import *

'''
# Check if a GPU is available and use it
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device:", torch.cuda.get_device_name(0))
# Check if on macOS and Metal is available
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
    print("Using Apple MPS (Metal Performance Shaders) on macOS")
# Fallback to CPU if no GPU or MPS
else:
    device = torch.device("cpu")
    print("Using CPU")
'''
print(torch.cuda.is_available())
device = torch.device("cuda")

import torch

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
                          TOKENIZER)
#%%
transformer
#%%
train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)

#%%
for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break

#%%
from torch import nn

criterion = nn.CrossEntropyLoss(ignore_index=it_vocabulary_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)

# A large negative constant used to represent negative infinity in mask calculations.
NEG_INFTY = -1e9


def create_masks(eng_batch, it_batch, tokenizer = None):
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
        if tokenizer:
            eng_sentence_length, it_sentence_length = len(tokenizer(eng_batch[idx])), len(tokenizer(it_batch[idx]))
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

transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, it_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, it_batch, TOKENIZER)
        optim.zero_grad()
        it_predictions = transformer(eng_batch,
                                     it_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(it_batch, start_token=False, end_token=True)
        loss = criterion(
            it_predictions.view(-1, it_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == it_vocabulary_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Italian Translation: {it_batch[0]}")
            it_sentence_predicted = torch.argmax(it_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in it_sentence_predicted:
              if idx == it_vocabulary_to_index[END_TOKEN]:
                break
              predicted_sentence += it_index_to_vocabulary[idx.item()]
            print(f"Italian Prediction: {predicted_sentence}")

            transformer.eval()
            it_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
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
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = it_index_to_vocabulary[next_token_index]
                it_sentence = (it_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break
            
            print(f"Evaluation translation (should we go to the mall?) : {it_sentence}")
            print("-------------------------------------------")

import os
model_save_path = os.path.join(MODEL_PATH,f"transformer_model_{TOKENIZATION_LEVEL}_level_tok.pth")
torch.save(transformer.state_dict(), model_save_path)


