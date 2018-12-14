###################################################################################
## Description: Run machine learning models on the data                         ###
## Status: WIP															        ###
###################################################################################
from sklearn.model_selection import KFold
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm
import pandas as pd
import numpy as np
import sys

## script settings
iteration = 1 													## UPDATE THIS: to improve data management

## path to data folder
path = '/home/irina/data'

if len(sys.argv) > 1:
    dataset = sys.argv[1] 
else:
    if not sys.stdin.isatty():
        inputfile = sys.stdin.read()
        if inputfile:
            inputfile = inputfile.split('\n')[0]
            if inputfile:
                dataset = inputfile
    else:
        dataset = path + '/dedoose.csv'
        print "Warning: Input file NOT specified for vectoriser script \nUsing default file " + dataset + "\n\nTo specify input file pipe processor script run: \npython 1_processRawData.py | 2_vectoriseBuildData.py\n"

## read data
try:
    df = pd.read_csv(dataset, encoding='utf-8')
except IOError:
    print "Fatal Error: Sorry, couldn't find " + dataset
    sys.exit(0)

y = df['Label']
X = df.drop(['Label','ID'], axis=1)


## k-fold cross validation
kf = KFold(n_splits=2, random_state=42, shuffle=True)
for train_index, test_index in kf.split(X):
    #print("TRAIN:", train_index, "TEST:", test_index)
    result = next(kf.split(df), None)
    train = df.iloc[result[0]]
    test =  df.iloc[result[1]]
    
    y_train = train['Label']
    X_train = train.drop(['Label','ID'], axis=1)

    y_test = test['Label']
    X_test = test.drop(['Label','ID'], axis=1)

    ## multinomial naive bayes 
    list_alpha = np.arange(0.01, 10.0, 0.1)
    score_train = np.zeros(len(list_alpha))
    score_test = np.zeros(len(list_alpha))
    recall_test = np.zeros(len(list_alpha))
    precision_test = np.zeros(len(list_alpha))
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
    print models.head(n=10)
    
    best_index = models['Test Accuracy'].idxmax()
    print models.iloc[best_index, :]

    m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
    print pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'],
                index = ['Actual 0', 'Actual 1'])


outputfile = ''

print outputfile

