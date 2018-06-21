###################################################################################
## Description: Vectorise build data                                            ###
## Status: WIP															        ###
###################################################################################
from nltk.corpus import stopwords
from sklearn import feature_extraction, preprocessing
import sys, time, re, string
import pandas as pd

## script settings
iteration = 1 													## UPDATE THIS: to improve data management

## path to data folder
path = '/home/irina/data'

if len(sys.argv) > 1:
    dataset = sys.argv[1]
    datacol = raw_input("What is the name of the DATA column? (press enter if not known)\n") 
    labelcol = raw_input("What is the name of the LABEL column? (press enter if not known) \n") 

    if datacol and labelcol:
        cols = [datacol,labelcol]
    else:
        cols = None
else:
    dataset = path + '/dedoose.csv'
    cols = ['Excerpt','Label'] 

## read data
df = pd.read_csv(dataset, usecols=cols, encoding='utf-8')

stop = set(stopwords.words('english')).union([''])

## Vectorise using counts
f = feature_extraction.text.CountVectorizer(preprocessor, stop_words='english', analyzer='word', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', binary=True) # automatically generates vocabulary 
X = f.fit_transform(df[cols[0]]) # generates counts
 
# print f.get_feature_names() # the list of features (unique words)
# print X.toarray() # the matrix showing the counts 

## Apply tf-idf term weighting
#f = feature_extraction.text.TfidfTransformer(smooth_idf=False)
#tfidf = f.fit_transform(X.toarray()) # generates weights of each feature

## Binarize
#X = preprocessing.binarize(X) # no need for counts, just a binary value representing existence of the word in the excerpt
#y = df['Label'].map({'egbotdont':1,'egbotdo':0}) # binarize labels as well


newX = pd.DataFrame(data=X.toarray().transpose(), index=f.get_feature_names())
vectorisedDF = newX.transpose().assign(**df.drop(['Excerpt'], axis=1)).copy()


## exporting dataset to csv
try:
    vectorisedDF.to_csv(path + '/2_vectoriseBuildData_%d_%d.csv' % (iteration, int(time.time())), sep=',', encoding='utf-8')
except UnicodeEncodeError as e: print(e)
