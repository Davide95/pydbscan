# -*- coding: utf-8 -*-

import numpy as np
from dbscan import DBSCAN as myDBSCANImpl
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans, DBSCAN

plt.figure(1)
features, labels = noisy_circles = datasets.make_circles(n_samples=1000, 
                                                         factor=.5, noise=.05)
kmeans = KMeans(n_clusters=2)  
kmeans.fit(features)  
plt.scatter(features[:,0], y=features[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title('K-means with k=2')
plt.show()

plt.figure(2)
features, labels = datasets.make_blobs(n_samples=1000, random_state=10)
kmeans = KMeans(n_clusters=2)  
kmeans.fit(features)
plt.scatter(features[:,0], y=features[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title('K-means with k=2')
plt.show()

plt.figure(3)
features, labels = datasets.make_blobs(n_samples=1000, random_state=2)
features[0] = [4, -7]
kmeans = KMeans(n_clusters=2)  
kmeans.fit(features)
plt.scatter(features[:,0], y=features[:,1], c=kmeans.labels_, cmap='rainbow')
plt.title('K-means with k=2')
plt.show()

plt.figure(4)
features, labels = datasets.make_blobs(n_samples=1000, random_state=10)
dbscan = myDBSCANImpl(epsilon=3, minPts=2) 
plt.scatter(features[:,0], y=features[:,1], c=dbscan.fit(features), cmap='rainbow')
plt.title('DBSCAN with ε too big and MinPts too small')
plt.show()

plt.figure(5)
centers = [[0.8, 0.8], [-0.8, -0.8], [0.8, -0.8]]
features, labels = datasets.make_blobs(n_samples=1000, centers=centers,
                                       cluster_std=0.4)
dbscan = myDBSCANImpl(epsilon=0.5, minPts=5) 
plt.scatter(features[:,0], y=features[:,1], c=dbscan.fit(features), cmap='rainbow')
plt.title('DBSCAN applied to a dataset with three clusters')
plt.show()

plt.figure(6)
features1, labels1 = datasets.make_gaussian_quantiles(mean=(0,0), cov=0.5, 
                                                    n_samples=300)
features2, labels2 = datasets.make_gaussian_quantiles(mean=(5,5), cov=5.0, 
                                                    n_samples=200)
features = np.append(np.array(features1), np.array(features2), axis=0)
dbscan = myDBSCANImpl(epsilon=0.5, minPts=3) 
plt.scatter(features[:,0], y=features[:,1], c=dbscan.fit(features), cmap='rainbow')
plt.title('A dataset with two cluster with different densities')
plt.show()