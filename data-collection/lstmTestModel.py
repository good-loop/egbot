from __future__ import print_function
from keras.models import load_model
import pickle
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None,
                    help="Path to trained model (default: dummy model)")
parser.add_argument('--vocab', type=str, default=None,
                    help="Path to vocab file (default: dummy vocab)")
parser.add_argument('--answer_size', type=int, default=30,
                    help='Desired generated answer size (default: 30)')
parser.add_argument('--seed', type=str, default='',
                    help='Desired seed for the model (default: empty string)')
args = parser.parse_args()

# the location of the model to load
if args.model:
    save_dir = args.model
else:
    file_list = ['math-sm'];
    save_dir = '../data/models/' + file_list[0] # directory to store models

seq_length = args.answer_size # sequence length
sequences_step = 1 #step to create sequences

# load vocabulary
print("loading vocabulary...")
with open(os.path.join(save_dir, "words_vocab.pkl"),'rb') as f:
    (words, vocab, vocabulary_inv) = pickle.load(f)

# vocabulary size
vocab_size = len(words)

# load the model
print("loading model...")
model = load_model(save_dir + "/" + 'gen_sentences_lstm_model.final.hdf5')

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#initiate sentences
if args.seed:
    seed_sentences = args.seed
else:
    seed_sentences = '' #"let us suppose that we toss up a penny a great many times the results of the successive throws may be conceived to form a "
#generated = ''
sentence = []
for i in range (seq_length):
    sentence.append("a")

seed = seed_sentences.split()

for i in range(len(seed)):
    sentence[seq_length-i-1]=seed[len(seed)-i-1]

#generatedStr += ' '.join(sentence)
print('Generating text with the following seed: "' + ' '.join(sentence[seq_length-len(seed):]) + '"')

print ()

# #tell it how many words it should generate
# if len(sys.argv) > 1:
#     words_number = int(args.answer_size)
# else:
#     words_number = 100

#generate the text
generated = seed
for i in range(seq_length):
    #create the vector
    x = np.zeros((1, seq_length, vocab_size))
    for t, word in enumerate(sentence):
        x[0, t, vocab[word]] = 1.
    #print(x.shape)

    #calculate next word
    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, 0.34)
    next_word = vocabulary_inv[next_index]

    #add the next word to the text
    #generatedStr += " " + next_word
    generated.append(next_word) 
    # shift the sentence by one, and and the next word at its end
    sentence = sentence[1:] + [next_word]

#print(generatedStr)
print('EgBot came up with this: ' + ' '.join(generated))

