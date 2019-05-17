import pandas as pd

datapath = "/home/irina/egbot-learning-depot/trained_glove_stanford/"
gloveFilename = "glove.6B.50d"
infile = datapath+gloveFilename+".txt"
outfile = datapath+gloveFilename+".pkl"

df = pd.read_csv(infile, sep=" ", quoting=3, header=None, index_col=0)
glove = {key: val.values for key, val in df.T.items()}

import pickle
with open(outfile, 'wb') as fp:
    pickle.dump(glove, fp)

print("Saved pkl file: "+outfile)
