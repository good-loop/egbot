from __future__ import print_function
from pprint import pprint
from time import time
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import neighbors
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

pipeline = Pipeline([
    ('vect', feature_extraction.text.CountVectorizer()),
    ('tfidf', feature_extraction.text.TfidfTransformer()),
    ('clf', neighbors.KNeighborsClassifier()),
])

parameters = {
    'vect__analyzer': ('char', 'word', 'char_wb'),
    'vect__stop_words': (None, 'english'),
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__smooth_idf': (True, False),
    'tfidf__sublinear_tf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__n_neighbors': np.arange(1, 10, 1),
    'clf__weights': ('uniform', 'distance'),
    'clf__algorithm': ('ball_tree', 'kd_tree', 'brute'),
    'clf__p': (1, 2)
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
    grid_search.fit(df['egbot_answer_body'], df['egbot_answer_label'])
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))