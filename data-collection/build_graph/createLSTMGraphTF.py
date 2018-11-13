# based on BiDirLSTM https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
# and saved in pb format https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py

# Just create an empty graph -- no training

import tensorflow as tf
import os

# Training Parameters
learning_rate = 0.001
#training_steps = 10000
#batch_size = 128
#display_step = 200
vocab_size = 116 # TODO: put in real number
num_hidden = 256 # hidden layer num of features
seq_length = 3

# check now that we're in the right place
egbotdir = os.path.abspath('../..')
assert egbotdir.endswith("egbot")

# tf Graph input
X = tf.placeholder(tf.float32, [None, seq_length, 1], name='input')
Y = tf.placeholder(tf.float32, [None, vocab_size], name='target')

# Define weights
weights = {
    # Hidden layer weights => 2*num_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, vocab_size], seed=42, mean=1.0), name='W')
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size], seed=24, mean=1.0), name='b')
}

def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # reshape to [1, seq_length]
    x = tf.reshape(x, [-1, seq_length])

    # Generate a seq_length-element sequence of inputs
    # (eg. [had] [a] [general] -> [20] [6] [33])
    x = tf.split(x, seq_length, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)
    #rnn_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_hidden),tf.nn.rnn_cell.BasicLSTMCell(num_hidden)])

    # Get lstm cell output
    outputs, _, _ = tf.nn.static_bidirectional_rnn (lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    #outputs, states = tf.nn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
#prediction = tf.identity(logits, name='output')
prediction = tf.nn.softmax(logits, name="output")

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y), name="loss_op")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name="optimizer")
train_op = optimizer.minimize(loss_op, name="train_op")

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name="correct_pred")
#correct = tf.nn.in_top_k(logits, Y, 1, name="correct_pred")
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
saver_def = tf.train.Saver().as_saver_def()

print('Operation to initialize variables:       ', init.name)
print('Tensor to feed as input data:            ', X.name)
print('Tensor to feed as training targets:      ', Y.name)
print('Tensor to fetch as prediction:           ', prediction.name)
print('Operation to train one step:             ', train_op.name)
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

sess = tf.Session()

with open('../../data/models/final/v3/lstmGraphTF.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())

# write to log c.f. https://stackoverflow.com/questions/37128652/creating-log-directory-in-tensorboard
summary_writer = tf.summary.FileWriter('../../data/models/final/v3/logdir', sess.graph)
# tf.train.SummaryWriter('../../data/models/final/v3/logdir', sess.graph_def)
# tf.get_default_graph().as_graph_def()) #

