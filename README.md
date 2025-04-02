# Transformer-based Neural Machine Translation with Different Tokenization Strategies

## Introduction
Transformers have significantly improved Neural Machine Translation (NMT) by removing the constraints of sequential processing, allowing for more efficient training and scalability to larger datasets. However, the performance of these models is highly sensitive to the tokenization strategy used, which breaks down text into tokens that the model can process. Tokenization directly impacts the model's input representation, influencing its ability to capture linguistic nuances and, consequently, its translation quality.

This project investigates three tokenization methods:
- **Character-level tokenization**: Treats each character as a separate token, capturing detailed information but resulting in longer sequences that complicate training.
- **Word-level tokenization**: Treats each word as an individual token, simplifying implementation but struggling with out-of-vocabulary (OOV) words and leading to a large vocabulary.
- **WordPiece tokenization**: A subword-level approach that balances vocabulary size and OOV handling by breaking words into smaller subwords.

We hypothesize that different tokenization strategies significantly impact the translation quality of Transformer-based Seq2Seq models, measured using the **Bilingual Evaluation Understudy (BLEU) score**. WordPiece tokenization is expected to offer the best trade-off between vocabulary size and translation performance.

## Dataset and Model
We use the **CCMatrix dataset**, which contains 32.7 billion unique sentences across 38 languages. For this project, we selected **203,000 English-Italian sentence pairs**, dividing them into **200,000 for training** and **3,000 for testing**.

The model is a **Seq2Seq Transformer** with:
- **Encoder-Decoder architecture** (1 layer each)
- **8 attention heads per layer**
- **Embedding size: 200**
- **Hidden size: 512**
- **ReLU activation functions**
- **Sinusoidal positional encodings**

## Experimental Setup
### Preprocessing
Sentences exceeding **200 characters** are excluded. Tokenization methods differ as follows:
- **Character-level**: 87-token vocabulary (alphabet, punctuation, and special tokens `<PADDING>`, `<START>`, `<END>`).
- **Word-level**: Vocabulary built from all unique words and punctuation, resulting in **59,885 English tokens** and **81,825 Italian tokens**.
- **WordPiece**: Subword tokenization with a vocabulary size of **10,000 tokens** per language, following the Hugging Face tutorial "Building a tokenizer, block by block".

### Training
The model is trained using the **Adam optimizer** to minimize **cross-entropy loss**, with:
- **Batch size: 30**
- **10 epochs**
- **Learning rate: 0.0001**
- **Dropout rate: 0.1**

Hyperparameters remain **constant** across all tokenization methods to ensure a fair comparison.

### Evaluation
Translation quality is assessed using the **BLEU score**, computed for the **3,000 test sentences** categorized by length:
- **Short (< 50 characters)**
- **Medium (50-100 characters)**
- **Long (> 100 characters)**

## Results
The BLEU scores for each tokenization method are summarized in the table below:

| Tokenization Level | S. length <= 50 | 50 < S. length <= 100 | S. length > 100 | Total |
|-------------------|---------------|----------------------|----------------|------|
| **Character**     | 0.16250       | 0.11960              | 0.11572        | 0.13260 |
| **Word**         | 0.00000       | 0.00000              | 0.0001         | 0.00000 |
| **WordPiece**    | 0.00021       | 0.00041              | 0.00008        | 0.00047 |

Character-level tokenization yielded the best performance, while WordPiece and word-level tokenization performed poorly.

## Discussion
Contrary to our hypothesis, **character-level tokenization outperformed other methods**. The model trained with word and WordPiece tokenization failed to learn translation effectively. A possible reason is that, with only **one layer in the encoder and decoder**, the model lacks sufficient capacity to handle large vocabularies.

### Future Work
Future experiments will investigate how tokenization strategies perform when:
- **Scaling the model size** (adding more encoder/decoder layers)
- **Extending the training duration**

By increasing model capacity and training time, we aim to determine when **WordPiece tokenization** begins to outperform character-level tokenization.

## References
- Bahdanau et al., "Neural Machine Translation by Jointly Learning to Align and Translate," 2016.
- Vaswani et al., "Attention is All You Need," 2017.
- Wu et al., "Google’s Neural Machine Translation System," 2016.
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," 2019.
- Papineni et al., "BLEU: A Method for Automatic Evaluation of Machine Translation," 2001.
- Schwenk et al., "CCMatrix: Mining Billions of High-Quality Parallel Sentences on the WEB," 2020.
refer to the documentation in the repository.
