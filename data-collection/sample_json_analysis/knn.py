from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn import model_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

# automatic feature extraction
#f = feature_extraction.text.CountVectorizer(analyzer='word', stop_words = 'english')#, vocabulary=feats)
#X = f.fit_transform(df['egbot_answer_body'])

#vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
#X = vectorizer.fit_transform(df['egbot_answer_body'])

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

# automatic feature extraction
vect = CountVectorizer(analyzer='word', stop_words=None, max_df=1.0, ngram_range=(1, 3), max_features=10000)#, vocabulary=feats)
X = vect.fit_transform(df['egbot_answer_body'])

tfidf = TfidfTransformer(norm='l1', use_idf=False, smooth_idf=True, sublinear_tf=True)
X = tfidf.fit_transform(X)
y = df['egbot_answer_label']

# plain train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.5, random_state=42)

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

# Y = vectorizer.transform(["take the following example. let's say that x is 0 and y is 1. so we can see that it is true."])
# Y = vectorizer.transform(["toma el siguiente ejemplo. digamos que x es 0 e y es 1. entonces podemos ver que es verdad"])
# prediction = model.predict(Y)
# print(prediction)

# n_neighbors = 15
# h = .02  # step size in the mesh

# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# for weights in ['uniform', 'distance']:
#     # we create an instance of Neighbours Classifier and fit the data.
#     clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
#     clf.fit(X_train, y_train)
#     score = clf.score(X_test, y_test)
#     print(score)

#     # Plot the decision boundary. For that, we will assign a color to each
#     # point in the mesh [x_min, x_max]x[y_min, y_max].
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                          np.arange(y_min, y_max, h))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#     # Put the result into a color plot
#     Z = Z.reshape(xx.shape)
#     plt.figure()
#     plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

#     # Plot also the training points
#     plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#                 edgecolor='k', s=20)
#     plt.xlim(xx.min(), xx.max())
#     plt.ylim(yy.min(), yy.max())
#     plt.title("2-Class classification (k = %i, weights = '%s')"
#               % (n_neighbors, weights))

# plt.show()

from sklearn import datasets

# Loading some example data
iris = datasets.load_iris()

from scipy.sparse import csr_matrix
X = csr_matrix(X).todense().A

X = X
y = y

n_neighbors = 15
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

fig, ax = plt.subplots()

clf = neighbors.KNeighborsClassifier(n_neighbors=5)
tt = 'KNN (k=5)'

clf.fit(X, y)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, alpha=0.4)
ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
ax.set_title(tt)

plt.show()