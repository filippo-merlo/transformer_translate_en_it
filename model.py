#%% ### TRNSFORMER MODEL IMPLEMENTATION ###
"from: https://github.com/ajhalthor/Transformer-Neural-Network/blob/main/transformer.py"

# Import necessary libraries
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

# Function to get the appropriate device (GPU if available, otherwise CPU)
def get_device():
    #return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    return torch.device('cpu')

# Function implementing scaled dot-product attention
def scaled_dot_product(q, k, v, mask=None):
    # Get the dimension of the keys
    d_k = q.size()[-1]
    # Compute the dot product of queries and transposed keys, scaled by sqrt(d_k)
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    
    # Apply mask if provided (used in attention mechanisms)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    
    # Compute softmax along the last dimension to get attention weights
    attention = F.softmax(scaled, dim=-1)
    # Multiply the attention weights with the values to get the final output
    values = torch.matmul(attention, v)
    
    return values, attention

# Class to implement positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        # Create position indices and denominators based on the dimension of the model
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        
        # Calculate the positional encodings using sine and cosine functions
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        
        # Stack and flatten to get the final positional encoding matrix
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

# Class to create sentence embeddings for input sentences
class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, tokenizer = None):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
        self.tokenizer = tokenizer
    
    # Function to tokenize and encode a batch of sentences
    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            # Convert each sentence into a list of token indices
            if self.tokenizer:
                for token in self.tokenizer(sentence):
                    print(token)
                sentence_word_indicies = [self.language_to_index[token] for token in self.tokenizer(sentence)]
            else:
                sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]
            # Add start and end tokens if specified
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            # Pad the sentence to match the maximum sequence length
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        # Apply tokenization to each sentence in the batch
        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append(tokenize(batch[sentence_num], start_token, end_token))
        
        # Stack all tokenized sentences into a single tensor and move to the appropriate device
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    # Forward method to compute sentence embeddings
    def forward(self, x, start_token, end_token): # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        print(x.shape)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x

# Class to implement multi-head self-attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model) # Linear layer for query, key, value projection
        self.linear_layer = nn.Linear(d_model, d_model) # Linear layer for the final output
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        # Compute query, key, and value vectors
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # Rearrange dimensions for multi-head attention
        q, k, v = qkv.chunk(3, dim=-1) # Split into query, key, and value tensors
        
        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask)
        
        # Combine the multiple heads and pass through a linear layer
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

# Class to implement layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        # Initialize gamma and beta parameters
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        # Compute mean and variance along the specified dimensions
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        
        # Compute the normalized output
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

# Class to implement the feed-forward network used in transformers
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden) # First linear layer
        self.linear2 = nn.Linear(hidden, d_model) # Second linear layer
        self.relu = nn.ReLU() # ReLU activation function
        self.dropout = nn.Dropout(p=drop_prob) # Dropout for regularization

    def forward(self, x):
        # Pass through the first linear layer, apply ReLU, dropout, and then the second linear layer
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# Class to implement a single encoder layer in the transformer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        # Define the components of the encoder layer
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        # Self-attention sub-layer
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        # Feed-forward sub-layer
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x

# Class to implement a sequence of encoder layers
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask) # Pass through each encoder layer sequentially
        return x

# Class to implement the full encoder of the transformer
class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN,
                 tokenizer = None):
        super().__init__()
        # Initialize the sentence embedding and encoder layers
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, tokenizer)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        # Create sentence embeddings and pass through the encoder layers
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x

# Class to implement multi-head cross attention (for decoder)
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model) # Linear layer for key-value projection
        self.q_layer = nn.Linear(d_model , d_model) # Linear layer for query projection
        self.linear_layer = nn.Linear(d_model, d_model) # Linear layer for final output
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # In practice, this is the same for both languages
        # Compute key-value vectors from one input and query vector from another
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3) # Rearrange dimensions for multi-head attention
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1) # Split into key and value tensors
        
        # Apply scaled dot-product attention
        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!
        
        # Combine the multiple heads and pass through a linear layer
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out

# Class to implement a single decoder layer in the transformer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        # Define the components of the decoder layer
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        # Self-attention sub-layer
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        # Encoder-decoder cross attention sub-layer
        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        # Feed-forward sub-layer
        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y

# Class to implement a sequence of decoder layers
class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask) # Pass through each decoder layer sequentially
        return y

# Class to implement the full decoder of the transformer
class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN,
                 tokenizer = None):
        super().__init__()
        # Initialize the sentence embedding and decoder layers
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, tokenizer)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        # Create sentence embeddings and pass through the decoder layers
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y

# Class to implement the full Transformer model (encoder + decoder)
class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                to_lang_vocab_size,
                from_lang_to_index,
                to_lang_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN,
                tokenizer
                ):
        super().__init__()
        # Initialize the encoder, decoder, and final linear layer
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, from_lang_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, tokenizer)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, to_lang_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN, tokenizer)
        self.linear = nn.Linear(d_model, to_lang_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Forward method to perform the entire encoding and decoding process
    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make this true
                dec_end_token=False): # x, y are batch of sentences
        
        # Encode the input sequence
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        
        # Decode the encoded sequence and generate the output sequence
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        
        # Pass the decoded output through the linear layer to get final logits
        out = self.linear(out)
        return out
