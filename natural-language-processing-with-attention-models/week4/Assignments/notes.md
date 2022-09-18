## Overview

Using the Reformer, also known as the efficient Transformer, to generate a dialogue between two bots.

For example, after a customer asks for a train ticket, the chatbot can ask what time the said customer wants to leave. You can use this concept to automate call centers, hotel receptions, personal trainers, or any type of customer service.

## Dataset

MultiWOZ -- A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling

1. more than 10,000 human annotated dialogues
2. spans multiple domains and topics.
3. number of conversations in the data set: 10438

### Content

**goal**<br>

The goal also points to a dictionary and it contains several keys pertaining to the objectives of the conversation. For example below, we can see that the conversation will be about booking a taxi.

```
DIALOGUE_DB['SNG0073.json']['goal']
```

```
{'taxi': {'info': {'leaveAt': '17:15',
   'destination': 'pizza hut fen ditton',
   'departure': "saint john's college"},
  'reqt': ['car type', 'phone'],
  'fail_info': {}},
 'police': {},
 'hospital': {},
 'hotel': {},
 'attraction': {},
 'train': {},
 'message': ["You want to book a <span class='emphasis'>taxi</span>. The taxi should go to <span class='emphasis'>pizza hut fen ditton</span> and should depart from <span class='emphasis'>saint john's college</span>",
  "The taxi should <span class='emphasis'>leave after 17:15</span>",
  "Make sure you get <span class='emphasis'>car type</span> and <span class='emphasis'>contact number</span>"],
 'restaurant': {}}
```

**log**<br>
The log on the other hand contains the dialog. It is a list of dictionaries and each element of this list contains several descriptions as well. Let's look at an example:

```
DIALOGUE_DB['SNG0073.json']['log'][0]
```

```
{'text': "I would like a taxi from Saint John's college to Pizza Hut Fen Ditton.",
 'metadata': {},
 'dialog_act': {'Taxi-Inform': [['Dest', 'pizza hut fen ditton'],
   ['Depart', "saint john 's college"]]},
 'span_info': [['Taxi-Inform', 'Dest', 'pizza hut fen ditton', 11, 14],
  ['Taxi-Inform', 'Depart', "saint john 's college", 6, 9]]}
```

### ref:

1. https://arxiv.org/abs/1810.00278
2. https://vimeo.com/306141298

## Training

1. convert person 1 and person 2 sentence as input

example1 :

```
Person 1: I would like a taxi from Saint John's college to Pizza Hut Fen Ditton. Person 2: What time do you want to leave and what time do you want to arrive by?
```

example 2

```
 Person 1: Hi. Can you help me find a train? Person 2: Where would you like to go to?
 Person 1: I will be going to Londons Kings Crossing.  Person 2: Sure, just let me get a little more information.  Where are you departing from and when did you want the booking for?  Person 1: I need to leave friday after 16:15 from cambridge. Person 2: TR6628 leaves Cambridge on Friday at 17:00 and arrives by 17:51 at london kings cross, does this suit your needs? Person 1: That sounds great, I'll need 4 seats please and a reference number. Person 2: I've booked that for you.  Your reference number is O4BRX03O and you'll owe 94.4 GBP payable at the station.  Anything else I can do? Person 1: Yes, please. I'm looking for an Italian restaurant in the east. Price range doesn't matter. Person 2: What area would you like to be in? Person 1: I want to eat in the East.  Person 2: Okay, there is one that matches. It's the Pizza Hut Fen Ditton. Would you like me to make a reservation for you? Person 1: Not at this time but can I please get their postcode? Person 2: Of course, the postcode is cb58wr. Is there anything else I can assist you with? Person 1: No, that will do it. Thank you and goodbye!  Person 2: Have a good visit.
```

2. tokenize the data to token id by using vocab_file

## Loss function

CrossEntropyLoss

## Output activations function

LogSoftmax

## Result

### Prediction

1. convert input sentence into token ids
2. call model
3. decode output tokens to word one by one

```
    # call the autoregressive_sample_stream function from trax
    output_gen = trax.supervised.decoding.autoregressive_sample_stream(
        # model
        ReformerLM,
        # inputs will be the tokens with batch dimension
        inputs=input_tokens_with_batch,
        # temperature
        temperature=temperature
```

We can now feed in different starting sentences and see how the model generates the dialogue. You can even input your own starting sentence. Just remember to ask a question that covers the topics in the Multiwoz dataset so you can generate a meaningful conversation.

## Reformer improvement

### Hashing attention (for time complexity)

1. Select a ramdon projection with several buckets
2. Project input vectors to the project
3. similiar vectors should be place into the same buckets
4. so instead of calculate all weighting in a sentence, just calculate the word with similar vector

### Reversible Transformer (for memory complexity)

1. calculate and using the same memory space
2. store the output layer y and the derivative of y
3. becase we can calcuate x from y, derivative of y and activation function, importantly this means we don't have to store x1 or x2!

## Idea

1. can use in automate call centers, but need to store data in a json/specific format
2. there is no label/y data, the model learn the weighting by observing the next word in training data
