import tensorflow as tf
import os
 
depot = '../../data/models/final/v3';
seq_length = 30
hm_epochs = 5
vocab_size = 15807

# load the graph
with tf.gfile.FastGFile(depot+'/lstmGraphTF.pb', 'rb') as f:
    print("Loading graph ...")
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    g_in = tf.import_graph_def(graph_def, name="")
    
save_dir = ''
with tf.Session(graph=g_in) as sess:
    print("Training ...")

    # tf Graph input
    # inp = sess.graph.get_tensor_by_name('input')  
    # out = sess.graph.get_tensor_by_name('target')  

    X = tf.placeholder(tf.float32, [None, seq_length, 1], name='inpu')
    Y = tf.placeholder(tf.float32, [None, vocab_size], name='targe')

    # input = tf.Variable(tf.random_normal([20, seq_length, 1], seed=42, mean=1.0), name='input')
    # target = tf.Variable(tf.random_normal([20, vocab_size], seed=42, mean=1.0), name='target')
 
    inp = [[[0 for _ in range(1)] for _ in range(seq_length)] for _ in range(1)]
    tar = [[0 for _ in range(vocab_size)] for _ in range(1)]

    #print(sess.run(y, {tf.get_default_graph().get_operation_by_name(input).outputs[0]: [1, 2, 3]}))

    i = 0
    for epoch in range(hm_epochs):
        epoch_loss = 0
        op = tf.get_default_graph().get_operation_by_name("train_op")
        sess.run(tf.global_variables_initializer())
        c = sess.run(op, feed_dict={'inpu:0': inp, 'targe:0': tar})
        epoch_loss += c

        print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

    # Evaluate model (with test logits, for dropout to be disabled)
    # correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1), name="correct_pred")
    # print('Correct_pred: ',correct_pred)

    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name="accuracy")
    # print('Accuracy: ',accuracy)

    #save the model
    modelPath = depot + '/test_lstm_model.pb'
    tf.saved_model.simple_save(
        sess, modelPath, inputs, outputs
    )

print("\nTa-Dah! Finished training.")
print("\nI saved the model here for you: " + os.path.abspath(depot))
