from __future__ import print_function
from sklearn import decomposition
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import sys
import os
import time
import codecs
import collections
import pickle
import spacy
nlp = spacy.load('en')
#nlp = spacy.blank('en')
#nlp.add_pipe(nlp.create_pipe('sentencizer'))

# device_name = sys.argv[1]

# if device_name == "gpu":
#     device_name = "/gpu:0"
# else:
#     device_name = "/cpu:0"

file_list = []
for i in [1,2,3,4,6]:#,6,7,8,9]:
    file_list.append("egbot_" + str(i))
print("List of files to extract: ", file_list)

#file_list = ['egbot'];
data_dir = '../data/egbot2/'# data directory 
save_dir = '../data/models/' + file_list[0] + '_' + str(int(time.time())) # directory to store models
os.mkdir(save_dir)
seq_length = 30 # sequence length
sequences_step = 1 #step to create sequences

vocab_file = os.path.join(save_dir, "words_vocab.pkl")

undesirables = ['\n', '\t', '\r']
markers = ['{', '}', '(', ')', '[', ']', '=', '+', '*', '/', '$']

def add_to_wordlist(word):
    if word:
        wl.append(text.lower())

def create_wordlist(doc):
    wl = []
    for word in doc:
        text = word.text 
        temp = None
        for undesired in undesirables:
            if undesired in text:
                split = text.split(undesired)
                #text = split[0] + split[1]
                temp = [split[0], split[1]]
        # for marker in markers:
        #     if marker in text:
        #         split = text.split(marker)
        #         temp = [split[0].lower(), marker, split[1].lower()]
        if not temp:
            temp = [text]
        wl = wl + temp
    return wl

wordlist = []
for file_name in file_list:
    print("Opening " + file_name)
    input_file = os.path.join(data_dir, file_name + ".txt")
    #read data
    with codecs.open(input_file, "r", "utf-8") as f:
        data = f.read()
    #create sentences
    doc = nlp(data)
    wl = create_wordlist(doc)
    wordlist = wordlist + wl

# count the number of words
word_counts = collections.Counter(wordlist)

# Mapping from index to word : that's the vocabulary
vocabulary_inv = [x[0] for x in word_counts.most_common()]
vocabulary_inv = list(sorted(vocabulary_inv))

# Mapping from word to index
vocab = {x: i for i, x in enumerate(vocabulary_inv)}
words = [x[0] for x in word_counts.most_common()]

#size of the vocabulary
vocab_size = len(words)
print("\nVocab size: ", vocab_size)

print("Saving vocabulary... ")
#save the words and vocabulary
with open(os.path.join(vocab_file), 'wb') as f:
    pickle.dump((words, vocab, vocabulary_inv), f)

#create sequences
sequences = []
next_words = []
for i in range(0, len(wordlist) - seq_length, sequences_step):
    sequences.append(wordlist[i: i + seq_length])
    next_words.append(wordlist[i + seq_length])

print('nb sequences:', len(sequences))

sequences = sequences[:1000001]
no_samples = len(sequences)
validation_split = 0.001
validation_cutoff = int(validation_split*no_samples)
print('Validation size: ' + str(validation_cutoff))

valid_seq = sequences[:validation_cutoff]
train_seq = sequences[validation_cutoff:]

#mem = 10000000.0
sz_batch = 64
#nb_batch = int(mem/(seq_length*vocab_size)) # quantity of seq that my ram can take
#sz_batch = #int(mem/(seq_length*vocab_size)) #int(len(sequences)/nb_batch)
nb_batch = int(len(train_seq)/sz_batch)
#sz_batch = int(len(train_seq)/nb_batch)

print('Batching data (size/number): ', sz_batch, nb_batch)

def valid_samples(seq, size):
    X = np.zeros((size, seq_length, vocab_size), dtype=np.bool)
    y = np.zeros((size, vocab_size), dtype=np.bool)
    for i, sentence in enumerate(seq):
        for t, word in enumerate(sentence):
            X[i, t, vocab[word]] = 1
        y[i, vocab[next_words[i]]] = 1
    return X, y

def train_samples(seq, size):
    while True:
        for start in range(0, len(seq), size):
            end = start + size
            if end > len(seq):
                end = len(seq)
            #print('Training with data from ' + str(start) + " to " + str(end))
            X = np.zeros((size, seq_length, vocab_size), dtype=np.bool)
            y = np.zeros((size, vocab_size), dtype=np.bool)
            for i, sentence in enumerate(seq[start:end]):
                for t, word in enumerate(sentence):
                    X[i, t, vocab[word]] = 1
                y[i, vocab[next_words[i]]] = 1
            yield X, y
    
# X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool)
# y = np.zeros((len(sequences), vocab_size), dtype=np.bool)
# for i, sentence in enumerate(sequences):
#     for t, word in enumerate(sentence):
#         X[i, t, vocab[word]] = 1
#     y[i, vocab[next_words[i]]] = 1

def bidirectional_lstm_model(seq_length, vocab_size):
    print('Build LSTM model.')
    model = Sequential()
    model.add(Bidirectional(LSTM(rnn_size, activation="relu"),input_shape=(seq_length, vocab_size)))
    model.add(Dropout(0.6))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    
    optimizer = Adam(lr=learning_rate)
    callbacks=[EarlyStopping(patience=2, monitor='val_loss')]
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[categorical_accuracy])
    return model

rnn_size = 16 # size of RNN
#batch_size = 32 # minibatch size
#seq_length = 20 # sequence length
num_epochs = 50 # number of epochs
learning_rate = 0.001 #learning rate
sequences_step = 1 #step to create sequences

md = bidirectional_lstm_model(seq_length, vocab_size)
md.summary()

#fit the model
callbacks=[EarlyStopping(patience=4, monitor='val_loss'),
           ModelCheckpoint(filepath=save_dir + "/" + 'gen_sentences_lstm_model.{epoch:02d}-{val_loss:.2f}.hdf5',\
                           monitor='val_loss', verbose=0, mode='auto', period=2)]
# history = md.fit(X, y,
#                  batch_size=batch_size,
#                  shuffle=True,
#                  epochs=num_epochs,
#                  callbacks=callbacks,
#                  validation_split=0.01)


with tf.device('/gpu:0'):
    history = md.fit_generator(
                        generator=train_samples(train_seq, sz_batch),
                        steps_per_epoch=nb_batch,
                        shuffle=True,
                        epochs=num_epochs,
                        callbacks=callbacks,
                        validation_data=valid_samples(valid_seq, len(valid_seq)),
                        validation_steps=1,
                        verbose=1)


with tf.Graph().as_default():
    with tf.Session as sess:

        inputs = {
            "batch_size": batch_size,
            "features": features,
            "labels": labels,
        }

        outputs = {"prediction": model_output}

        #save the model
        modelPath = save_dir + '/gen_sentences_lstm_model.final.pb'
        tf.saved_model.simple_save(
            sess, modelPath, inputs, outputs
        )

print("\nTa-Dah! Finished training.")
print("\nI saved the model here for you: " + os.path.abspath(save_dir))
