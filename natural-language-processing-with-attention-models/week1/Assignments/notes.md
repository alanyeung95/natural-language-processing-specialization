## Overview

LSTM + attention(Q,K,V)

## Input

VOCAB_FILE = 'ende_32k.subword'

```
'<pad>_'
'<EOS>_'
', _'
'._'
'the_'
'_'
'in_'
'of_'
'and_'
'to_'
'die_'
```

```
THIS IS THE ENGLISH SENTENCE:
 Contact your doctor immediately if you become pregnant, think you might be pregnant or are planning to become pregnant while taking LYRICA.


THIS IS THE TOKENIZED VERSION OF THE ENGLISH SENTENCE:
  [21758   139  8937  2626   175    72   449  3678 17363     2   597    72
   616    32  3678 17363    66    31  3376     9   449  3678 17363   459
   981  2474  4318 31318   176  3550 30650  4729   992     1     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0]

THIS IS THE GERMAN TRANSLATION:
 Suchen Sie sofort Ihren Arzt auf, wenn Sie während der Behandlung mit LYRICA schwanger werden, glauben schwanger zu sein oder eine Schwangerschaft planen.


THIS IS THE TOKENIZED VERSION OF THE GERMAN TRANSLATION:
 [15775    23    67  5210  1786 32806    37     2   157    67   408    11
  3544    39  2474  4318 31318   176 16718 16989    58     2  3294 16718
 16989    18   171    97    41 21145  3393  2121 11011  3550 30650  4729
   992     1     0     0     0     0     0     0     0     0     0     0
     0     0     0     0     0     0     0     0     0     0     0     0
     0     0     0     0]
```

## Model

### Input encoder

The input encoder runs on the input tokens, creates its embeddings, and feeds it to an LSTM network. This outputs the activations that will be the keys and values for attention.

```
# UNQ_C1
# GRADED FUNCTION
def input_encoder_fn(input_vocab_size, d_model, n_encoder_layers):
    """ Input encoder runs on the input sentence and creates
    activations that will be the keys and values for attention.

    Args:
        input_vocab_size: int: vocab size of the input
        d_model: int:  depth of embedding (n_units in the LSTM cell)
        n_encoder_layers: int: number of LSTM layers in the encoder
    Returns:
        tl.Serial: The input encoder
    """

    # create a serial network
    input_encoder = tl.Serial(

        ### START CODE HERE (REPLACE INSTANCES OF `None` WITH YOUR CODE) ###
        # create an embedding layer to convert tokens to vectors
        tl.Embedding(input_vocab_size, d_model),

        # feed the embeddings to the LSTM layers. It is a stack of n_encoder_layers LSTM layers
        [ tl.LSTM(d_model) for _ in range(n_encoder_layers) ]
        ### END CODE HERE ###
    )

    return input_encoder
```

### Pre-attention decoder

The pre-attention decoder runs on the targets and creates activations that are used as queries in attention.

Append start tag like <SOS> in the beginning

```
# UNQ_C2
# GRADED FUNCTION
def pre_attention_decoder_fn(mode, target_vocab_size, d_model):
    """ Pre-attention decoder runs on the targets and creates
    activations that are used as queries in attention.

    Args:
        mode: str: 'train' or 'eval'
        target_vocab_size: int: vocab size of the target
        d_model: int:  depth of embedding (n_units in the LSTM cell)
    Returns:
        tl.Serial: The pre-attention decoder
    """

    # create a serial network
    pre_attention_decoder = tl.Serial(

        ### START CODE HERE (REPLACE INSTANCES OF `None` WITH YOUR CODE) ###
        # shift right to insert start-of-sentence token and implement
        # teacher forcing during training
        tl.ShiftRight(1, mode),

        # run an embedding layer to convert tokens to vectors
        tl.Embedding(target_vocab_size, d_model),

        # feed to an LSTM layer
        tl.LSTM(d_model)
        ### END CODE HERE ###
    )

    return pre_attention_decoder
```

### Whole model

```
# UNQ_C4
# GRADED FUNCTION
def NMTAttn(input_vocab_size=33300,
            target_vocab_size=33300,
            d_model=1024,
            n_encoder_layers=2,
            n_decoder_layers=2,
            n_attention_heads=4,
            attention_dropout=0.0,
            mode='train'):
    """Returns an LSTM sequence-to-sequence model with attention.

    The input to the model is a pair (input tokens, target tokens), e.g.,
    an English sentence (tokenized) and its translation into German (tokenized).

    Args:
    input_vocab_size: int: vocab size of the input
    target_vocab_size: int: vocab size of the target
    d_model: int:  depth of embedding (n_units in the LSTM cell)
    n_encoder_layers: int: number of LSTM layers in the encoder
    n_decoder_layers: int: number of LSTM layers in the decoder after attention
    n_attention_heads: int: number of attention heads
    attention_dropout: float, dropout for the attention layer
    mode: str: 'train', 'eval' or 'predict', predict mode is for fast inference

    Returns:
    A LSTM sequence-to-sequence model with attention.
    """

    ### START CODE HERE (REPLACE INSTANCES OF `None` WITH YOUR CODE) ###

    # Step 0: call the helper function to create layers for the input encoder
    input_encoder = input_encoder_fn(input_vocab_size, d_model, n_encoder_layers)

    # Step 0: call the helper function to create layers for the pre-attention decoder
    pre_attention_decoder = pre_attention_decoder_fn(mode, target_vocab_size, d_model)

    # Step 1: create a serial network
    model = tl.Serial(

      # Step 2: copy input tokens and target tokens as they will be needed later.
      tl.Select([0,1,0,1]),

      # Step 3: run input encoder on the input and pre-attention decoder the target.
      tl.Parallel(input_encoder, pre_attention_decoder),

      # Step 4: prepare queries, keys, values and mask for attention.
      tl.Fn('PrepareAttentionInput', prepare_attention_input, n_out=4),

      # Step 5: run the AttentionQKV layer
      # nest it inside a Residual layer to add to the pre-attention decoder activations(i.e. queries)
      tl.Residual(tl.AttentionQKV(d_model, n_heads=n_attention_heads, dropout=attention_dropout, mode=mode)),

      # Step 6: drop attention mask (i.e. index = None
      tl.Select([0,2]),

      # Step 7: run the rest of the RNN decoder
      [tl.LSTM(d_model) for _ in range(n_decoder_layers)],

      # Step 8: prepare output by making it the right size
      tl.Dense(target_vocab_size),

      # Step 9: Log-softmax for output
      tl.LogSoftmax()
    )

    ### END CODE HERE

    return model
```
