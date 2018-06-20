from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction
import string
import re
import pandas as pd
import numpy as np
import operator
import sklearn

# path to data folder
path = '/home/irina/data'

#========== reading data 
df = pd.read_csv(path + '/dedoose_smaller.csv', usecols=['Excerpt','Label'], encoding='utf-8')   # Index Excerpt Label

feats = []
stop = set(stopwords.words('english'))
word_freq = [] #df.shape[0]*[0]

for index, row in df.iterrows():    # Index Word1 Word2 ... WordN Label
    words = re.findall(r"[\w']+", row['Excerpt'].lower())
    words_filtered = dict()
    for word in words:
        #word = re.sub('[^A-Za-z0-9]+', '', word)
        if word[0] == '\'' and word[-1] == '\'':
            word = word[1:-1]
        elif word[0] == '_' and word != '_':
            word = word[1:]
        if word != '' and word not in stop and word not in string.punctuation and not bool(re.match('^[0-9]+$', word)):
            
            if word not in feats:
                feats.append(word)
            if word in words_filtered:
                words_filtered[word] += 1
            else:
                words_filtered[word] = 1

    word_freq.append(words_filtered)

#df1 = pd.DataFrame(data=dict.fromkeys(feats,0), index=range(0,df.shape[0])).assign(**df.drop(['Excerpt'], axis=1)).copy()

y = dict.fromkeys(feats,[0]*df.shape[0])

# add counts for each word for each row
for index, row in df.iterrows():   
    for label in word_freq[index].keys():
        #    print index, label, word_freq[index][label]
        y[label][index] = word_freq[index][label]
        #if int(word_freq[index][label]) > 1:
            #print index, label, word_freq[index][label], df1.loc[index][label]


df1 = pd.DataFrame(data=y, index=range(0,df.shape[0])).assign(**df.drop(['Excerpt'], axis=1)).copy()


# automatic feature extraction
# f = feature_extraction.text.CountVectorizer(analyzer='word', stop_words = 'english')#, vocabulary=feats)
# X = f.fit_transform(df['Excerpt'])
