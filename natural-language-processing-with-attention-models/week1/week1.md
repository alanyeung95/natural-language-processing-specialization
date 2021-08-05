## Dataset
Opus (available on Tensorflow Datasets(TFDS))

## Translation
ende_32k.subword with both english and german words
using same index to represent both english and german sentence

##  Attention Overview

![image](https://user-images.githubusercontent.com/45751387/120888687-45646480-c62c-11eb-9ff1-23b4fa62d139.png)

RNN is week on very long sentences (e.g. 100 tokens or more) because the context of the first parts of the input will have very little effect on the final vector passed to the decoder.

the attention layer will first receive all the encoder hidden states, it will score each of the encoder hidden states to know which one the decoder should focus on to produce the next word

Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

## LogSoftmax
LogSoftmax acts on a group of values and normalizes them to look like a set of log probability values. (Probability values must be non-negative, and as a set must sum to 1. A group of log probability values can be seen as the natural logarithm function applied to a set of probability values.)
