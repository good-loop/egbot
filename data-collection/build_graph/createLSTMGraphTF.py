# based on BiDirLSTM https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/bidirectional_rnn.py
# and saved in pb format https://github.com/tensorflow/models/blob/master/samples/languages/java/training/model/create_graph.py

import tensorflow as tf

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200
vocab_size = 100000 # TODO: put in real number
num_hidden = 128 # hidden layer num of features
seq_length = 30

# tf Graph input
X = tf.placeholder("float", [None, seq_length, 1])
Y = tf.placeholder(tf.float32, [None, vocab_size], name='target') # training target

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*num_hidden, vocab_size]), name='W')
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]), name='b')
}

def BiRNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, num_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, num_input)
    x = tf.unstack(x, seq_length, 1)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)
    # Backward direction cell
    lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_hidden, forget_bias=1.0, activation=tf.nn.relu)

    # Get lstm cell output
    outputs, _, _ = tf.nn.static_bidirectional_rnn (lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2 (
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
print('Tensor to feed as input data:            ', X.name)
print('Tensor to feed as training targets:      ', Y.name)
print('Tensor to fetch as prediction:           ', prediction.name)
print('Operation to train one step:             ', train_op.name)
print('Tensor to be fed for checkpoint filename:', saver_def.filename_tensor_name)
print('Operation to save a checkpoint:          ', saver_def.save_tensor_name)
print('Operation to restore a checkpoint:       ', saver_def.restore_op_name)
print('Tensor to read value of W                ', weights['out'].value().name)
print('Tensor to read value of b                ', biases['out'].value().name)

with open('lstmGraphTF.pb', 'wb') as f:
  f.write(tf.get_default_graph().as_graph_def().SerializeToString())
