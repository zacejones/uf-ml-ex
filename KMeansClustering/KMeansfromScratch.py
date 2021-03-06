#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 17:22:51 2021

@author: zacejones
"""

import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib import style
style.use('ggplot')

from matplotlib import colors

X = np.array([[1,2],
              [1.5,1.8],
              [5,8],
              [8,8],
              [1,0.6],
             [9,11]])
             
plt.scatter(X[:,0], X[:,1], s=150)
plt.show()

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        
    def fit(self,data):
        
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
            
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
                
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                
                self.classifications[classification].append(featureset)

                prev_centroids = dict(self.centroids)
                
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                
            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0)>self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
                    
                if optimized:
                    break
                
        def predict(self,data):
            distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            return classification
        
model = K_Means()
model.fit(X)

for centroid in model.centroids:
    plt.scatter(model.centroids[centroid][0], model.centroids[centroid][1], marker="o", color="k", s=150, linewidths=5)
            
for classification in model.classifications:
   r = random.random()
   b = random.random()
   g = random.random()
   color = (r, g, b)
   for featureset in model.classifications[classification]:
       plt.scatter(featureset[0],featureset[1], marker="x", color=color, s=150, linewidths=5)
        
plt.show()



