from nltk.corpus import stopwords
from wordcloud import WordCloud
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import Binarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import string
import re
import pandas as pd
import numpy as np
import operator
import sklearn

# path to data folder
path = '/home/irina/winterwell/egbot/data/raw'

#========== reading data 
df = pd.DataFrame()
cols = ['egbot_answer_body','egbot_answer_id','egbot_answer_label'] 
try:
    with open(path + '/d127+labelled.json', 'r') as read_file:
        df = pd.read_json(read_file, encoding='utf-8')
    df = df[cols]
except IOError:
    print 'Fatal Error: Sorry, couldn\'t find the dataset'
    sys.exit(0)

# automatic feature extraction
f = feature_extraction.text.CountVectorizer(analyzer='word', stop_words = 'english')#, vocabulary=feats)
X = f.fit_transform(df['egbot_answer_body'])

# # binarize
df['egbot_answer_label']=df['egbot_answer_label'].map({'TRUE':1,'FALSE':0})

# plain train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['egbot_answer_label'], test_size=0.5, random_state=42)

# multinomial naive bayes 
list_alpha = np.arange(0.01, 1.0, 0.01)
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1 

matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
models = pd.DataFrame(data = matrix, columns = 
             ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
 
best_index = models['Test Accuracy'].idxmax()
print models.iloc[best_index, :]

bayes = naive_bayes.MultinomialNB(alpha=models.iloc[best_index, :]['alpha'])
bayes.fit(X_train, y_train)
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test), labels=[1,0])
print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 1', 'Predicted 0'],
            index = ['Actual 1', 'Actual 0'])

### tasks todo
# what do the confusing spans look like?
# dimensonality reduction => pca or tsne
# knn results? similar to nb results? the nb results seem too good
# confusion matrix check
# marker words

## kmeans
# from sklearn.cluster import KMeans

# # automatic feature extraction
# #f = feature_extraction.text.CountVectorizer(analyzer='word', stop_words = 'english')#, vocabulary=feats)
# #X = f.fit_transform(df['egbot_answer_body'])

# vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
# X = vectorizer.fit_transform(df['egbot_answer_body'])

# # # binarize
# df['egbot_answer_label']=df['egbot_answer_label'].map({'TRUE':1,'FALSE':0})

# # plain train-test split
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['egbot_answer_label'], test_size=0.5, random_state=42)

# true_k = 2
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X_train)

# print("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     print

# print("\n")
# print("Prediction")

# for idx, val in enumerate(X_test):
#     if y_test[idx] == val:

# m_confusion_test = metrics.confusion_matrix(y_test, model.predict(X_test), labels=[1,0])
# print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 1', 'Predicted 0'],
#            index = ['Actual 1', 'Actual 0'])

pipeline = Pipeline([
    ('vect', feature_extraction.text.CountVectorizer()),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf', KNeighborsClassifier())
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__norm': (None, 'l1', 'l2'),
    'clf__n_neighbors': np.arange(1, 10, 1),
    'clf__weights': ('uniform', 'distance'),
    'clf__algorithm': ('ball_tree', 'kd_tree', 'brute'),
    'clf__p': (1, 2)
}

from pprint import pprint
from time import time
import logging

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(df['egbot_answer_body'], df['egbot_answer_label'])
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))



#Y = vectorizer.transform(["take the following example. let's say that x is 0 and y is 1. so we can see that it is true."])
#Y = vectorizer.transform(["toma el siguiente ejemplo. digamos que x es 0 e y es 1. entonces podemos ver que es verdad"])
# prediction = model.predict(Y)
# print(prediction)


# # svm
# list_C = np.arange(500, 2000, 100) #100000
# score_train = np.zeros(len(list_C))
# score_test = np.zeros(len(list_C))
# recall_test = np.zeros(len(list_C))
# precision_test= np.zeros(len(list_C))
# count = 0
# for C in list_C:
#     svc = svm.SVC(C=C)
#     svc.fit(X_train, y_train)
#     score_train[count] = svc.score(X_train, y_train)
#     score_test[count]= svc.score(X_test, y_test)
#     recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
#     precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
#     count = count + 1 

# matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
# models = pd.DataFrame(data = matrix, columns = 
#              ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
# print models.head(n=10)

# best_index = models['Test Accuracy'].idxmax()
# print models.iloc[best_index, :]

# m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
# print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])















#for index, row in df.iterrows():
#    words = row['Excerpt'].lower().split(' ')
#    for word in words:
#        df.loc[0,] = [1,2,3]


# #========== constructing frequency dictionaries
# for index, row in df.iterrows():
#     words = row['Excerpt Copy'].lower().split(' ')
#     label = row['Meta Label']
   
#     for word in words:
#         if label in 'egbotdont':
#             if word in egbotdont:
#                 egbotdont[word] += 1
#             else:
#                 egbotdont[word] = 1
#         if label in 'egbotdo':
#             if word in egbotdo:
#                 egbotdo[word] += 1
#             else:
#                 egbotdo[word] = 1

# #=========== word cloud

# # remove stopwords => nltk 
# stop = set(stopwords.words('english'))

# egbotdo_filtered = dict()
# egbotdont_filtered = dict()

# for word in egbotdo:
#     if word not in stop:
#         egbotdo_filtered[word] = egbotdo[word]
# for word in egbotdont:
#     if word not in stop:
#         egbotdont_filtered[word] = egbotdont[word]


#========== feature extraction

# dataframes feat and label

# bag of words model - term frequency
# matrix = []
# #matrix.append(egbotdo)
# #matrix.append(egbotdont)
# v = DictVectorizer(sparse=False)
# X = v.fit_transform(matrix)
# print v.get_feature_names()


#unique = list(set(freq_do+freq_dont))
#bowdo = [0]*len(unique)
##bowdont = [0]*len(unique)

#print unique
# for word in egbotdont:
#     print word
#     bowdont[unique.index(word)] = egbotdont[word]
#     break
        # if label in 'egbotdo':
        #     bowdo[unique.index(word)] += 1


# later: phrase based features, part of speech features, semantic features, social cues, etc?
# named entity recog: rule-based, generative models - HMM, conditional models - MeMM or CRF, deep learning - RNN?

#========== clustering 

#========== rule-based

#========== classification 
# w/ and w/o stemming
# try knn: bayes, svm, decision trees, random forests 

# naive bayes


#========== testing

#=========== general improvements
# should i use tf-idf to normalise terms?
# n gram model?
# needs more data => input some mse data as well






