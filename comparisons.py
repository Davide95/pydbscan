# -*- coding: utf-8 -*-

from dbscan import DBSCAN as myDBSCANImpl
import matplotlib.pyplot as plt
import sklearn.cluster
from sklearn import datasets

def comparison(features: np.ndarray, figure: int, epsilon: float, minPts: int):
    """It compares my implementation with the Scikit-learn implementation of DBSCAN."""
    plt.figure(figure)
    
    plt.subplot(1, 2, 1)
    dbscanIns = sklearn.cluster.DBSCAN(eps=epsilon, min_samples=minPts)
    dbscanIns.fit(features)
    plt.scatter(features[:,0], y=features[:,1], c=dbscanIns.labels_, cmap='rainbow')
    plt.title('Scikit-learn implementation')
    plt.show()
    
    plt.subplot(1, 2, 2)
    myDbscan = myDBSCANImpl(epsilon=epsilon, minPts=minPts)
    labels = myDbscan.fit(features)
    plt.scatter(features[:,0], y=features[:,1], c=labels, cmap='rainbow')
    plt.title('My implementation')
    plt.show()
    

features1, labels1 = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)
comparison(features1, 1, 0.1, 3)

features2, labels2 = sklearn.datasets.make_moons(n_samples=1000, noise=.05)
comparison(features2, 2, 0.1, 3)

features3, labels3 = datasets.make_blobs(n_samples=1000, random_state=10)
comparison(features3, 3, 1, 3)

