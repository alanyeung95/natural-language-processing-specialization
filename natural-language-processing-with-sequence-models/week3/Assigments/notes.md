## Overview

We first start by defining named entity recognition (NER). NER is a subtask of information extraction that locates and classifies named entities in a text. The named entities could be organizations, persons, locations, times, etc.

```
geo: geographical entity
org: organization
per: person
gpe: geopolitical entity
tim: time indicator
art: artifact
eve: event
nat: natural phenomenon
O: filler word
```

## Sample Data

Using data from Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners.

data:

```
SENTENCE: Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country .
```

label:

```
SENTENCE LABEL: O O O O O O B-geo O O O O O B-geo O O O O O B-gpe O O O O O
```

\* Everything else that is labeled with an O is not considered to be a named entity.

## Preprocessing

1. When training an LSTM using batches, all your input sentences must be the same size. To accomplish this, you set the length of your sentences to a certain number and add the generic `<PAD>` token to fill all the empty spaces.

2. convert word to token id and label to label id

sample output

```
The number of outputs is tag_map 17
Num of vocabulary words: 35181
The vocab size is 35181
The training size is 33570
The validation size is 7194
An example of the first sentence is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 9, 15, 1, 16, 17, 18, 19, 20, 21]
An example of its corresponding label is [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]
So you can see that we have already encoded each sentence into a tensor by converting it into a number. We also have 16 possible classes, as shown in the tag map.
```

sample tag

```
{'O': 0, 'B-geo': 1, 'B-gpe': 2, 'B-per': 3, 'I-geo': 4, 'B-org': 5, 'I-org': 6, 'B-tim': 7, 'B-art': 8, 'I-art': 9, 'I-per': 10, 'I-gpe': 11, 'I-tim': 12, 'B-nat': 13, 'B-eve': 14, 'I-eve': 15, 'I-nat': 16}
```

## Training

1. using LSTM, word ordering will affect the NER. For example, Peter eat cake have pattern Person, Verb, Object

### Model Structure

```
def NER(vocab_size=35181, d_model=50, tags=tag_map):
    '''
      Input:
        vocab_size - integer containing the size of the vocabulary
        d_model - integer describing the embedding size
      Output:
        model - a trax serial model
    '''
    ### START CODE HERE (Replace instances of 'None' with your code) ###
    model = tl.Serial(
      tl.Embedding(vocab_size, d_model), # Embedding layer
      tl.LSTM(d_model), # LSTM layer
      tl.Dense(len(tags)), # Dense layer with len(tags) units
      tl.LogSoftmax()  # LogSoftmax layer
      )
      ### END CODE HERE ###
    return model
```

### Loss function

CrossEntropyLoss

## Result

```
Step    100: train CrossEntropyLoss |  0.61237383
Step    100: eval  CrossEntropyLoss |  0.36804661
Step    100: eval          Accuracy |  0.91130057
```

## Q&A

### why logSoftmax vs softmax?

Log Softmax is advantageous over softmax for numerical stability, optimisation and heavy penalisation for highly incorrect class.

e.g.

```
Probability
[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Log Prob :
[-2.3  -1.61 -1.2  -0.92 -0.69 -0.51 -0.36 -0.22 -0.11  0. ]
```
