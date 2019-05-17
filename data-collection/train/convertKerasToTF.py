import tensorflow as tf
import keras.models
import keras.backend
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
sess = tf.Session()

print('\nSetting up model...')
g = tf.get_default_graph() 
with g.device('/device:GPU:0'):
  # Initializing the variables
  model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

init = tf.global_variables_initializer() 

# save as ckpt
saver = tf.train.Saver()
model = keras.models.load_model("model.h5")
sess = keras.backend.get_session()
save_path = saver.save(sess, "model.ckpt")