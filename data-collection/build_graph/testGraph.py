# based on BiDirLSTM https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
# and saved in pb format https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py

# Just create an empty graph -- no training
from random import randint
import tensorflow as tf
import os

# Training Parameters
learning_rate = [0.001]
#training_steps = 10000
#batch_size = 128
#display_step = 200
vocab_size = 5 #15807 #13346 #1197 # TODO: put in real number
num_hidden = 256#128 #256 # hidden layer num of features
seq_length = 3

# check now that we're in the right place
egbotdir = os.path.abspath('../..')
assert egbotdir.endswith("egbot"), egbotdir+" Try running from the build_graph dir"

def BiRNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data inp shape: (batch_size, timesteps, n_inp)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_inp)

    # reshape to [1, seq_length]
    x = tf.reshape(x, [-1, seq_length])

    # Generate a seq_length-element sequence of inps
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, seq_length, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)
    #rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_hidden),tf.nn.rnn_cell.BasicLSTMCell(num_hidden)])

    # Get lstm cell output
    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    #outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

hm_epochs = 100
dataSize = 10

## random data
# inp = [[[randint(0, 4) for _ in range(1)] for _ in range(seq_length)] for _ in range(dataSize)]
# print(inp)
# tar = [[0 for _ in range(vocab_size)] for _ in range(dataSize)]
# for i in range(len(tar)):
#     tar[i][randint(0, seq_length)] = 1
# print(tar)

## data with pattern
inp = [[[1], [2], [3]], [[3], [2], [1]], [[1], [2], [1]], [[3], [2], [1]], [[3], [1], [2]], [[1], [2], [3]], [[2], [1], [3]], [[1], [1], [2]], [[1], [2], [2]], [[1], [2], [4]]]
tar = [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]]

def train_neural_network(rate):
    print("Learning rate: ", rate)
    g = tf.Graph()
    with g.device('/device:GPU:0'):
        logits = BiRNN(X, weights, biases)
        #prediction = tf.identity(logits, name='output')    
        prediction = tf.nn.softmax(logits, name="output")

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y), name="loss_op")
        optimizer = tf.train.AdamOptimizer(learning_rate=rate, name="optimizer")
        train_op = optimizer.minimize(loss_op, name="train_op")

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for i in range(dataSize):
                    _, c = sess.run([train_op,loss_op], feed_dict={'inp:0': inp[i], 'target:0': tar[i]})
                    epoch_loss += c
                if(epoch%(hm_epochs/10)==0):
                    print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            # Evaluate model (with test logits, for dropout to be disabled)
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y), name="correct_pred")
            accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'), name="accuracy")

            # #print('Correct_pred: ',correct_pred)
            avg_acc = 0
            for i in range(len(tar)):
                acc = accuracy.eval({'inp:0': inp[i], 'target:0': tar[i]})
                # if(acc < 1):
                #     print(i)
                avg_acc += acc
            print('Avg Training Accuracy: ', avg_acc/len(tar))


for rate in learning_rate:

    # tf Graph inp
    X = tf.placeholder(tf.float32, [seq_length, 1], name='inp')
    Y = tf.placeholder(tf.float32, [vocab_size], name='target')

    # Define weights
    weights = {
        # Hidden layer weights => 2*num_hidden because of forward + backward cells
        'out': tf.Variable(tf.random_normal([2*num_hidden, vocab_size], seed=42, mean=1.0), name='W')
    }
    biases = {
        'out': tf.Variable(tf.random_normal([vocab_size], seed=24, mean=1.0), name='b')
    }

    train_neural_network(rate);

    tf.reset_default_graph();

print('Done :)');




