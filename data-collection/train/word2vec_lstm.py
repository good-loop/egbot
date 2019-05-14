from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
from keras.backend import get_session
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import tensorflow as tf
import numpy as np
import gensim
import string
import os  

def sample(preds, temperature=1.0):
  if temperature <= 0:
    return np.argmax(preds)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)

def generate_next(text, num_generated=10):
  word_idxs = [word2idx(word) for word in text.lower().split()]
  for i in range(num_generated):
    prediction = model.predict(x=np.array(word_idxs))
    idx = sample(prediction[-1], temperature=0.7)
    word_idxs.append(idx)
  return ' '.join(idx2word(idx) for idx in word_idxs)

def on_epoch_end(epoch, _):
  print('\nGenerating text after epoch: %d' % epoch)
  texts = [
    'deep convolutional',
    'simple and effective',
    'a nonconvex',
    'a',
  ]
  for text in texts:
    sample = generate_next(text)
    print('%s... -> %s' % (text, sample))


print('\nFetching the text...')
#url = 'https://raw.githubusercontent.com/maxim5/stanford-tensorflow-tutorials/master/data/arxiv_abstracts.txt'
#path = get_file('arxiv_abstracts.txt', origin=url)
path = "/home/irina/winterwell/egbot/data/test_input/egbotTextFiles/egbot.txt"

max_sentence_len = 40
batch_size = 100000

print('\nPreparing the sentences...')
with open(path) as file_:
  docs = file_.readlines()
sentences = [[word for word in doc.lower().translate(str.maketrans('','',string.punctuation)).split()[:max_sentence_len]] for doc in docs]
print('Num sentences:', len(sentences))

sentences = sentences[0:batch_size]

fileExists = os.path.isfile('word2vec.model')
if fileExists is True: 
  print('\nLoading trained word2vec model...')
  word_model = Word2Vec.load('word2vec.model')
  pretrained_weights = word_model.wv.syn0
  vocab_size, emdedding_size = pretrained_weights.shape
  print('Result embedding shape:', pretrained_weights.shape)
  print('Checking similar words:')
  # for word in ['model', 'network', 'train', 'learn']:
  #     most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  #     print('  %s -> %s' % (word, most_similar))
else:
  print('\nTraining word2vec model...')
  word_model = Word2Vec(sentences, size=100, min_count=1, window=5, iter=100)
  pretrained_weights = word_model.wv.syn0
  vocab_size, emdedding_size = pretrained_weights.shape
  print('Result embedding shape:', pretrained_weights.shape)
  # print('Checking similar words:')
  # for word in ['model', 'network', 'train', 'learn']:
  #     most_similar = ', '.join('%s (%.2f)' % (similar, dist) for similar, dist in word_model.most_similar(word)[:8])
  #     print('  %s -> %s' % (word, most_similar))
  print('\nSaving word2vec...')
  word_model.save('word2vec.model')

def word2idx(word):
  return word_model.wv.vocab[word].index
def idx2word(idx):
  return word_model.wv.index2word[idx]

print('\nPreparing the data for LSTM...')
train_x = np.zeros([len(sentences), max_sentence_len], dtype=np.int32)
train_y = np.zeros([len(sentences)], dtype=np.int32)
for i, sentence in enumerate(sentences):
  if sentence:
    for t, word in enumerate(sentence[:-1]):
        train_x[i, t] = word2idx(word)
    last = sentence[-1]
    train_y[i] = word2idx(last)
print('train_x shape:', train_x.shape)
print('train_y shape:', train_y.shape)

print('\nConfiguring session...')
sess = tf.Session()
sess.as_default()
set_session(sess)  # set this TensorFlow session as the default session for Keras

print('\nSetting up model...')
g = tf.get_default_graph() 
with g.device('/device:GPU:0'):
  model = Sequential()

model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights]))
model.add(LSTM(units=emdedding_size))
model.add(Dense(units=vocab_size))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print("Vocab size: " + str(vocab_size))
print("Embed size: " + str(emdedding_size))

# Initializing the variables
init = tf.global_variables_initializer() 

fileExists = os.path.isfile('model.h5')
if False: 
  model = load_model("model.h5")
  sess = get_session()
else:
  print('\nTraining LSTM...')
  # fit the model
  model.fit(train_x, train_y,
      batch_size=128,
      epochs=100)

  # evaluate the model
  # scores = model.evaluate(X, Y, verbose=0)
  # print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  
  # serialize model to JSON
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
      json_file.write(model_json)
  
  # serialize weights to HDF5
  model.save_weights("model.h5")
  print("Saved model to disk")

# save as ckpt
saver = tf.train.Saver()
save_path = saver.save(sess, "./model.ckpt")