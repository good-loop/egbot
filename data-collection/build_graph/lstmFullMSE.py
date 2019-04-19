# training lstm with full MSE data (using a size-limited vocab)
#
# author: Irina

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from collections import Counter
import time
import json
import os
import nltk

nltk.download('punkt') # needs a download of nltk lib to use tokeniser

#import spacy
#nlp = spacy.load('en')

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

# Vocabulary size limit
vocabLimit = 100000

# Training files
datapath = '/home/irina/winterwell/egbot/data/build/slim/';
file_list = []
for i in [1]:#,2,3,4,5,6,7,8]:
    #file_list.append(datapath + 'MathStackExchangeAPI_Part_' + str(i) + '.json')
    #training_file = '/home/irina/winterwell/egbot/data/test_input/belling_the_cat.json'
    #file_list.append(training_file)
    training_file = '/home/irina/winterwell/egbot/data/test_input/pauliusSample.json'
    file_list.append(training_file)
print("List of files to extract: ", file_list)

# using collections.Counter() because we only want to keep most common words (to prevent memory leaks), limit is set by vocabLimit
vocabCount = Counter();

def read_data(fname):
    print("\nLoading file: " + fname)
    with open(fname) as json_file:
        data = json.load(json_file)
    return data

def processJSONFile(data):
    qa = ""
    # concatenate question and answer
    for elem in data:
        qa += elem["question"] + " " + elem["answer"] + " "

    # ensure correct encoding
    qa = str(bytes(qa, encoding='utf-8'),encoding="ascii",errors='ignore')

    # remove white leading/trailing white spaces and lowercase it
    text = qa.strip().lower()
    print("Tokenising...")

    # tokenise it
    training_data = nltk.word_tokenize(text)

    print("Finished tokenising")
    return training_data

def build_vocab():
    for trainFile in file_list:
        training_data = read_data(trainFile)   
        words = processJSONFile(training_data)
        count = Counter(words)
        vocabCount.update(count);

print("\nBuilding vocab...")
build_vocab()

vocab_size = len(vocabCount.keys())
print("\nFull vocab size: " + str(vocab_size))

vocabCount = dict(vocabCount.most_common(vocabLimit))
vocab_size = len(vocabCount)
print("Trimmed vocab size: " + str(vocab_size))
print()

# construct vocab and reverse_vocab as mappings from words to unique id number and vice-versa
vocab = dict()
for idx, word in enumerate(vocabCount.keys()):
    vocab[word] = idx
vocab["<UNKNOWN>"] = idx + 1 # only useful if we had to trim the vocab for memory reasons
vocab["<START>"] = idx + 2
reverse_vocab = dict(zip(vocab.values(), vocab.keys())) # useful for reverse look-up

# Parameters
learning_rate = 0.0001 
training_iters = 500
#training_iters = 500000
display_step = 100
seq_length = 30

# number of units in RNN cell
n_hidden = 512

# Experiment
experimentName = "lstmPythonGraphTF_vocab=" + str(vocab_size) + "_numHidden=" + str(n_hidden) + "_seqLength=" + str(seq_length) + "_trainingIters=" + str(training_iters) + ".pb";
experimentGraphFile = '../../data/models/final/v3/'+experimentName;

# tf Graph input
x = tf.placeholder("float", [None, seq_length, 1])
y = tf.placeholder("float", [None, vocab_size+2])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size+2]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size+2]))
}

def RNN(x, weights, biases):
    # reshape to [1, seq_length]
    x = tf.reshape(x, [-1, seq_length])

    # Generate a seq_length-element sequence of inputs
    # (eg. [What] [if] [we] -> [20] [6] [33])
    x = tf.split(x,seq_length,1)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are seq_length outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

g = tf.get_default_graph() 
with g.device('/device:GPU:0'):
    pred = RNN(x, weights, biases)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

def checkIfValidWord(idx, training_data):
    if training_data[idx] in vocab.keys():
        return vocab[str(training_data[idx])]
    else:
        return vocab["<UNKNOWN>"]

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    #    pos = random.randint(0,seq_length+1)
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    for trainFile in file_list:
        # Load qa data from file as list of words e.g. ["What", "if", "we", etc.]
        training_data = processJSONFile(read_data(trainFile))

        print("Starting training iteration...")
        while step < training_iters:
            pos = 0        
            while pos != len(training_data)-seq_length:        
                symbols_in_keys = []
                if pos < seq_length-1:
                    for i in range(0,seq_length-pos-1):
                        symbols_in_keys.append(vocab["<START>"])
                    loc = 0
                    for idx in range(seq_length-pos-1,seq_length):
                        symbols_in_keys.append(checkIfValidWord(idx, training_data))
                        loc += 1
                else:
                    for idx in range(pos-seq_length+1,pos+1):
                        symbols_in_keys.append(checkIfValidWord(idx, training_data))
                symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, seq_length, 1])
                symbols_out_onehot = np.zeros([vocab_size+2], dtype=float)
                symbols_out_onehot[checkIfValidWord(pos+1, training_data)] = 1.0
                symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])
                
                _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                        feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                loss_total += loss
                acc_total += acc
                if (step+1) % display_step == 0:
                    print("Iter= " + str(step+1) + ", Average Loss= " + \
                        "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                        "{:.2f}%".format(100*acc_total/display_step))
                    print("Elapsed time: ", elapsed(time.time() - start_time))
                    print()
                    acc_total = 0
                    loss_total = 0
                    symbols_in = [training_data[i] for i in range(pos, pos + seq_length)]
                    symbols_out = training_data[pos + seq_length]
                    symbols_out_pred = reverse_vocab[int(tf.argmax(onehot_pred, 1).eval())]
                    print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
                step += 1
                pos += 1
    print("\nOptimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your browser to: http://localhost:6006/")
    for i in range(3):
        prompt = "%s words: " % seq_length
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        symbols_in_keys = []
        if len(words) < seq_length:
            for i in range(0,seq_length-1):
                symbols_in_keys.append(vocab["<START>"])
        for i in range(len(words)):
            symbols_in_keys.append(vocab[str(words[i])])
        if len(words) > seq_length:
            for i in range(len(words)-seq_length,len(words)):
                symbols_in_keys.append(vocab[str(words[i])])
        try:
            for i in range(seq_length):
                keys = np.reshape(np.array(symbols_in_keys), [-1, seq_length, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_vocab[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in vocab")

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
saver_def = tf.train.Saver().as_saver_def()

with open(experimentGraphFile, 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

print("\nI saved the graph here for you: " + os.path.abspath(experimentGraphFile))
