from nltk.corpus import stopwords
from wordcloud import WordCloud
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
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

# plain train-test split
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['Label'], test_size=0.33, random_state=42)

# # binarize
# df['Label']=df['Label'].map({'egbotdont':1,'egbotdo':0})
# y = df['Label']

# k-fold cross validation
# kf = KFold(n_splits=2, random_state=42)
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]



# # multinomial naive bayes 
# list_alpha = np.arange(0.00001, 10.0, 0.001)
# score_train = np.zeros(len(list_alpha))
# score_test = np.zeros(len(list_alpha))
# recall_test = np.zeros(len(list_alpha))
# precision_test= np.zeros(len(list_alpha))
# count = 0
# for alpha in list_alpha:
#     bayes = naive_bayes.MultinomialNB(alpha=alpha)
#     bayes.fit(X_train, y_train)
#     score_train[count] = bayes.score(X_train, y_train)
#     score_test[count]= bayes.score(X_test, y_test)
#     recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
#     precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
#     count = count + 1 

# matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
# models = pd.DataFrame(data = matrix, columns = 
#              ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
# #print models.head(n=10)
 
# best_index = models['Test Accuracy'].idxmax()
# print models.iloc[best_index, :]

# m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
# print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
#             index = ['Actual 0', 'Actual 1'])

# # kmeans
# from sklearn.cluster import KMeans

# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(df['Excerpt'])

# true_k = 2
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)

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

# Y = vectorizer.transform(["take the following example. let's say that x is 0 and y is 1. so we can see that it is true."])
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






