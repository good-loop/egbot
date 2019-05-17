# based on BiDirLSTM https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
# and saved in pb format https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py
# Just creates an empty graph -- no training

import tensorflow as tf
import os
from tensorflow.contrib import rnn

# check now that we're in the right place
egbotdir = os.path.abspath('../..')
assert egbotdir.endswith("egbot"), egbotdir+" Try running from the build_graph dir"

# Training Parameters
learning_rate = 0.001
#batch_size = 128 # should we batch? it might be tricky to sync this with the java code

# WARNING: this might trip you up, because it needs the vocab size of the training data; TODO: this can be solved by having the script run with a parameter that tells it what the vocab size is expected to be (this is useful for the code to be able to run for any training data)
vocab_size = 1197
num_hidden = 256 # number of units in RNN cell
seq_length = 10 # length of training window

# tf Graph input
x = tf.placeholder("float", [None, seq_length, 1], name='input')
y = tf.placeholder("float", [None, vocab_size], name='target')

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, vocab_size]), name="W")
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]), name="b")
}

def twoLayerLSTM(x, weights, biases):

    # reshape to [1, seq_length]
    x = tf.reshape(x, [-1, seq_length])

    # Generate a seq_length-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, seq_length, 1)
    words = tf.identity(x, name="words")

    # define model to be a 2-layer LSTM, each layer has num_hidden units 
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(num_hidden),rnn.BasicLSTMCell(num_hidden)])

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are seq_length outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

g = tf.get_default_graph() 

with g.device('/device:GPU:0'):
    raw_pred = tf.identity(twoLayerLSTM(x, weights, biases), name="twoLayerLSTM"); # raw prediction
    activation1 = tf.nn.relu(raw_pred, name="relu"); # negative values turned to 0 (because negative means deactivated)
    activation2 = tf.nn.softmax(activation1, name="softmax");
    output = tf.identity(activation2, name="output");

    # Loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y), name="loss_op")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
    train_op = optimizer.minimize(loss_op, name="train_op")

    # Model evaluation
    correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(y,1), name="correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

# Initializing the variables
init = tf.global_variables_initializer() 

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
saver_def = tf.train.Saver().as_saver_def()

print('Operation to initialize variables:       ', init.name)
print('Tensor to feed as input data:            ', x.name)
print('Tensor to feed as training target:       ', y.name)
print('Operation to train one step:             ', train_op.name)
print('Tensor to fetch raw logits:              ', raw_pred.name)
print('Tensor to fetch activation1 outputs:     ', activation1.name)
print('Tensor to fetch activation2 outputs:     ', activation2.name)
print('Tensor to fetch prediction outputs:      ', output.name)
print('Tensor to fetch prediction accuracy:     ', correct_pred.name)
print('Tensor to fetch training accuracy:       ', accuracy.name)
print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
print('Tensor to read value of W                ', weights['out'].value().name)
print('Tensor to read value of b                ', biases['out'].value().name)

def mkdir(path):
	try:
		os.mkdir(path)
	except:
		print("meh")

mkdir('../../data/models')
mkdir('../../data/models/final')
mkdir('../../data/models/final/v3')
mkdir('../../data/models/final/v3/logdir')

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# set path name with experiment info
experimentName = "lstmGraphTF_vocab=" + str(vocab_size) + "_numHidden=" + str(num_hidden) + "_seqLength=" + str(seq_length) 
experimentName += "_modelType=" + str(raw_pred.name) + "_activation1=" + str(activation1.name) + "_activation2=" + str(activation2.name) + ".pb";
experimentGraphFile = '../../data/models/final/v3/'+experimentName;

with open(experimentGraphFile, 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

# write to log c.f. https://stackoverflow.com/questions/37128652/creating-log-directory-in-tensorboard
summary_writer = tf.summary.FileWriter('../../data/models/final/v3/logdir', sess.graph)
# tf.train.SummaryWriter('../../data/models/final/v3/logdir', sess.graph_def)
# tf.get_default_graph().as_graph_def()) #

print("\nI saved the graph here for you: " + os.path.abspath(experimentGraphFile))

