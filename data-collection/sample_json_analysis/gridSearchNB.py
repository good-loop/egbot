from __future__ import print_function
from pprint import pprint
from time import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
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
    print('Fatal Error: Sorry, couldn\'t find the dataset')
    sys.exit(0)

# # binarize
df['egbot_answer_label']=df['egbot_answer_label'].map({'TRUE':1,'FALSE':0})

# plain train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(df['egbot_answer_body'], df['egbot_answer_label'], test_size=0.1, random_state=42)

pipeline = Pipeline([
    ('vect', feature_extraction.text.CountVectorizer()),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf', naive_bayes.MultinomialNB()),
])

parameters = {
#   'vect__analyzer': ('char', 'word', 'char_wb'),
    'vect__analyzer': ('char', 'word'),   
#   'vect__stop_words': (None, 'english'),
   'vect__stop_words': (None, 'english'),
#   'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_df': (0.5, 0.75),
#   'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
   'vect__ngram_range': ((1, 1), (1, 2)), 
   'tfidf__use_idf': (True, False),
   'tfidf__smooth_idf': (True, False),
   'tfidf__sublinear_tf': (True, False),
   'tfidf__norm': ('l1', 'l2'),
#   'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0],
   'clf__alpha': [0.01, 0.05],
   'clf__fit_prior': (True, False)
}

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
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# automatic feature extraction
vect = feature_extraction.text.CountVectorizer(analyzer=best_parameters['vect__analyzer'], stop_words=best_parameters['vect__stop_words'], max_df=best_parameters['vect__max_df'], ngram_range=best_parameters['vect__ngram_range'])#, vocabulary=feats)
X = vect.fit_transform(df['egbot_answer_body'])

tfidf = feature_extraction.text.TfidfTransformer(norm=best_parameters['tfidf__norm'], use_idf=best_parameters['tfidf__use_idf'], smooth_idf=best_parameters['tfidf__smooth_idf'], sublinear_tf=best_parameters['tfidf__sublinear_tf'])
X = tfidf.fit_transform(X)

# plain train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, df['egbot_answer_label'], test_size=0.1, random_state=42)

# multinomial naive bayes 
list_alpha = [best_parameters['clf__alpha']]
score_train = np.zeros(len(list_alpha))
score_test = np.zeros(len(list_alpha))
recall_test = np.zeros(len(list_alpha))
precision_test= np.zeros(len(list_alpha))
count = 0
for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=best_parameters['clf__alpha'], fit_prior=best_parameters['clf__fit_prior'])
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
print(models.iloc[best_index, :])

bayes = naive_bayes.MultinomialNB(alpha=best_parameters['clf__alpha'], fit_prior=best_parameters['clf__fit_prior'])
bayes.fit(X_train, y_train)
m_confusion_test = metrics.confusion_matrix(y_true=y_test, y_pred=bayes.predict(X_test), labels=[1,0])
print(pd.DataFrame(data = m_confusion_test, columns = ['Predicted 1', 'Predicted 0'],
            index = ['Actual 1', 'Actual 0']))