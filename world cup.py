from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pickle
import numpy as np

features = [[20,11],[20,5],[20,12],[17,7],[16,7],[18,7],[19,7],[20,4],[20,9],[20,10]]
#labels = [[0],[1],[1],[0],[1],[0],[1],[0],[0],[1]]
labels = [0,1,1,0,1,0,1,0,0,1]
X = np.array(features)
y = np.array(labels)
clf = NearestCentroid()
clf.fit(X, y)
print(clf.predict([[20, 7]]))

print(clf.predict([[7, 20]]))

list_pickle = open('lr.pkl', 'wb')
pickle.dump(clf, list_pickle)
