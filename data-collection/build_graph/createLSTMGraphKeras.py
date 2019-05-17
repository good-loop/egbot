import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

#TODO: put in Keras code, then save keras model in pb format https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128 
display_step = 200 # frequence of training updates

# Network Parameters
seq_length = 30 # sequence length
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder(tf.float32, [None, seq_length, 1], name='input')
y = tf.placeholder(tf.float32, [None, vocab_size], name='target')

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def BiRNN(x, weights, biases):
    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, activation=tf.relu)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, activation=tf.relu)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        sequence_length=seq_length, dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        sequence_length=seq_length, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Creating a tf.train.Saver adds operations to the graph to save and
# restore variables from checkpoints.
saver_def = tf.train.Saver().as_saver_def()

print('Operation to initialize variables:       ', init.name)
print('Tensor to feed as input data:            ', x.name)
print('Tensor to feed as training targets:      ', y_.name)
print('Tensor to fetch as prediction:           ', y.name)
print('Operation to train one step:             ', train_op.name)
print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
print('Tensor to read value of W                ', W.value().name)
print('Tensor to read value of b                ', b.value().name)

with open('graph.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
