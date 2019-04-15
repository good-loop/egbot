from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from collections import Counter
import time
import json
import os

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
vocabLimit = 100

# Training files
datapath = '/home/irina/winterwell/egbot/data/build/slim/';
file_list = []
for i in [1]: #,2,3,4,5,6,7,8]:
    file_list.append(datapath + 'MathStackExchangeAPI_Part_' + str(i) + '.json')
print("List of files to extract: ", file_list)

# using collections.Counter() because we only want to keep most common words (to prevent memory leaks), limit is set by vocabLimit
vocabCount = Counter();

def read_data(fname):
    print("Loading file: " + fname)
    with open(fname) as json_file:
        data = json.load(json_file)
    return data

def processJSONFile(data):
    qa = ""
    for elem in data:
        qa += elem["question"] + " " + elem["answer"] + " "
    return qa.strip()

def build_vocab():
    for trainFile in file_list:
        training_data = read_data(trainFile)   
        words = processJSONFile(training_data).split()
        count = Counter(words)
        vocabCount.update(count);

print("\nBuilding vocab...")
build_vocab()

vocab_size = len(vocabCount.keys())
print("\nFull vocab size: " + str(vocab_size))

vocabCount = dict(vocabCount.most_common(vocabLimit))
vocab_size = len(vocabCount)
print("Trimmed vocab size: " + str(vocab_size))

# construct vocab and reverse_vocab as mappings from words to unique id number and vice-versa
vocab = dict()
for idx, word in enumerate(vocabCount.keys()):
    vocab[word] = idx
reverse_vocab = dict(zip(vocab.values(), vocab.keys())) # useful for reverse look-up

# Parameters
learning_rate = 0.001 
training_iters = 500
display_step = 1000 
n_input = 3 # sequence length

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):
    # reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # Generate a n_input-element sequence of inputs
    # (eg. [What] [if] [we] -> [20] [6] [33])
    x = tf.split(x,n_input,1)

    # 2-layer LSTM, each layer has n_hidden units.
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
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

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        for trainFile in file_list:
            # Load qa data from file as list of words e.g. ["What", "if", "we", etc.]
            training_data = processJSONFile(read_data(trainFile)).split()

            # Generate a minibatch. Add some randomness on selection process.
            if offset > (len(training_data)-end_offset):
                offset = random.randint(0, n_input+1)

            for i in range(offset, offset+n_input):
                if training_data[i] in vocab.keys(): # if the word appears in the vocab
                    symbols_in_keys = vocab[ str(training_data[i])]
                    symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])

                    symbols_out_onehot = np.zeros([vocab_size], dtype=float)
                    symbols_out_onehot[vocab[str(training_data[offset+n_input])]] = 1.0
                    symbols_out_onehot = np.reshape(symbols_out_onehot,[1,-1])

                    _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                            feed_dict={x: symbols_in_keys, y: symbols_out_onehot})
                    loss_total += loss
                    acc_total += acc
                    if (step+1) % display_step == 0:
                        print("Iter= " + str(step+1) + ", Average Loss= " + \
                            "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                            "{:.2f}%".format(100*acc_total/display_step))
                        acc_total = 0
                        loss_total = 0
                        symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
                        symbols_out = training_data[offset + n_input]
                        symbols_out_pred = reverse_vocab[int(tf.argmax(onehot_pred, 1).eval())]
                        print("%s - [%s] vs [%s]" % (symbols_in,symbols_out,symbols_out_pred))
                    step += 1
                    offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your browser to: http://localhost:6006/")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        try:
            symbols_in_keys = [vocab[str(words[i])] for i in range(len(words))]
            for i in range(32):
                keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
                onehot_pred = session.run(pred, feed_dict={x: keys})
                onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
                sentence = "%s %s" % (sentence,reverse_vocab[onehot_pred_index])
                symbols_in_keys = symbols_in_keys[1:]
                symbols_in_keys.append(onehot_pred_index)
            print(sentence)
        except:
            print("Word not in vocab")

