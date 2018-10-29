
# script that loads trained model and vocabulary and generates a sentence based on a seed sentence (adapted for integration with JEP)
# @author irina @zero-point

# import libraries
import sys
import tensorflow
import tensorflow.python
from keras.models import load_model
import pickle
import numpy as np
import argparse
import os

modelVersion = 'v2'
load_dir = 'data/models/final/' + modelVersion # directory where models are stored
seq_length = 30 # sequence length (depends on how the model was trained)
sequences_step = 1 #step to create sequences

# load vocabulary
with open(os.path.join(load_dir, "words_vocab.pkl"),'rb') as f:
    (words, vocab, vocabulary_inv) = pickle.load(f)

# vocabulary size
vocab_size = len(words)

# load the model
model = load_model(load_dir + "/" + 'gen_sentences_lstm_model.final.hdf5')


# helper function to sample an index from a probability array
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# method to generate sentence based on seed sentence
def generateResults(seed_sentences=''):
    sentence = []
    for i in range (seq_length):
        sentence.append("a")

    seed = seed_sentences.split()

    for i in range(len(seed)):
        sentence[seq_length-i-1]=seed[len(seed)-i-1]

    #generate the text
    generated = [] #seed
    for i in range(seq_length):
        #create the vector
        x = np.zeros((1, seq_length, vocab_size))
        for t, word in enumerate(sentence):
            if word in vocab.keys():
                x[0, t, vocab[word]] = 1.

        #calculate next word
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, 0.34)
        next_word = vocabulary_inv[next_index]

        #add the next word to the text
        generated.append(next_word) 
        # shift the sentence by one, and and the next word at its end
        sentence = sentence[1:] + [next_word]

    return(' '.join(generated))
