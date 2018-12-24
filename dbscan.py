# -*- coding: utf-8 -*-

import numpy as np

class DBSCAN:
    not_visited = 0 # The label of all not yet visited points
    noise = 1 # The label of points marked as noise
    
    def __init__(self, epsilon: float, minPts: int):
        self._epsilon = epsilon
        self._minPts = minPts
        
    def _getLabel(self, index: int) -> int:
        """It returns the cluster number associated with the point at position `index`."""
        return self._labels[index]
    
    def _setLabel(self, index: int, value: int):
        """It sets the label of the point at position `index` with value `value`."""
        self._labels[index] = value
        
    def _getPoint(self, index: int):
        """It returns the point at the position `index`."""
        return self._X[index]
        
    def distance(Q, P) -> np.float64:
        """It computes the distance between P and Q"""
        o = np.subtract(Q, P)
        o = np.power(o, 2)
        o = np.sum(o)
        o = np.sqrt(o)
        return o # return sqrt((q1-p1)^2 + ... + (qn-pn)^2)
    
    
    def _getNeighbours(self, P_index: int):
        """It returns all the points near the point P"""
        P = self._getPoint(P_index)

        neighbours = []
        for i in range(self._X.shape[0]):
            Q = self._getPoint(i)
            dist = DBSCAN.distance(P, Q)
            if dist <= self._epsilon:
                neighbours.append(i)

        return np.array(neighbours)
    

    def _expand(self, neighboursIndexes: np.ndarray, cluster: int):
        """It iterates through neighbours of neighbours"""
        for i in neighboursIndexes:
            if self._getLabel(i) == DBSCAN.not_visited:
                self._setLabel(i, cluster)
                newNeighboursIndexes = self._getNeighbours(i)
                if len(newNeighboursIndexes) >= self._minPts:
                    self._expand(newNeighboursIndexes, cluster)
        
    
    def fit(self, X: np.ndarray) -> np.ndarray:
        """It runs the algorithm on the dataset X"""
        self._X = X
        N = X.shape[0]
        self._labels = np.zeros(N)
        
        cluster = 2
        for i in range(N):
            if self._getLabel(i) == DBSCAN.not_visited:
                neighboursIndexes = self._getNeighbours(i)
                if len(neighboursIndexes) < self._minPts:
                    self._setLabel(i, DBSCAN.noise) # Mark the point as noise
                else:
                    self._setLabel(i, cluster)
                    self._expand(neighboursIndexes, cluster)
                    cluster += 1
        
        return self._labels