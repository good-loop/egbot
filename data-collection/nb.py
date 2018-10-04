from nltk.corpus import stopwords
from wordcloud import WordCloud
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.preprocessing import Binarizer
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

# # binarize
df['egbot_answer_label']=df['egbot_answer_label'].map({'TRUE':1,'FALSE':0})

# automatic feature extraction
vect = feature_extraction.text.CountVectorizer(analyzer='word', stop_words=None, max_df=0.75, max_features=None, ngram_range=(1, 2))#, vocabulary=feats)
X = vect.fit_transform(df['egbot_answer_body'])

tfidf = feature_extraction.text.TfidfTransformer(norm='l1', use_idf=False, smooth_idf=True, sublinear_tf=True)
X = tfidf.fit_transform(X)

# plain train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['egbot_answer_label'], test_size=0.5, random_state=42)

# multinomial naive bayes 
list_alpha = [0.01]
fit_prior = False

score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha, fit_prior=fit_prior)
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

bayes = naive_bayes.MultinomialNB(alpha=models.iloc[best_index, :]['alpha'], fit_prior=fit_prior)
bayes.fit(X_train, y_train)
m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test), labels=[1,0])
print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 1', 'Predicted 0'],
            index = ['Actual 1', 'Actual 0'])
