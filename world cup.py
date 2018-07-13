from sklearn import tree
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
import pickle
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

features = [[20,11],[20,5],[20,12],[17,7],[16,7],[18,7],[19,7],[20,4],[20,9],[20,10]]
r_fea = [[a[1],a[0] ]for a in features]
#labels = [[0],[1],[1],[0],[1],[0],[1],[0],[0],[1]]
labels = [0,1,1,0,1,0,1,0,0,1]
r_lab = [(a-1)*(a-1) for a in labels]
X = np.array(features+r_fea)
y = np.array(labels + r_lab)
clf = NearestCentroid()
clf.fit(X, y)
print(clf.centroids_)
print(clf.score(X,y))
print(clf.predict([[20, 7]]))

print(clf.predict([[7, 20]]))

list_pickle = open('lr.pkl', 'wb')
pickle.dump(clf, list_pickle)

cmap_light = ListedColormap(['#FFAAAA',  '#AAAAFF'])
h = .02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
loss = X[np.where(y == 0)]
win = X[np.where(y == 1)]
plt.axis([0, 22, 0, 22])
plt.plot([a[0] for a in loss], [a[1] for a in loss] , "o")
plt.plot([a[0] for a in win], [a[1] for a in win] , ".")
#plt.plot([a[0] for a in clf.centroids_], [a[1] for a in clf.centroids_], "^")
plt.plot([20], [7], "^")
plt.show()
