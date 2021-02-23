# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 08:07:36 2021

@author: s.ruffini
"""


from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


def silhouette_score_(k_rng, tfidf_matrix):
    
    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    sil = []
    for k in k_rng :
      kmeans = KMeans(n_clusters = k).fit(tfidf_matrix)
      labels = kmeans.predict(tfidf_matrix)
      sil.append(silhouette_score(tfidf_matrix, labels))
    
    print(len(sil))
    #plot
    plt.plot(k_rng,np.array(sil))
    plt.xlabel("k")
    plt.ylabel("Silhouette index")
    plt.show()
    plt.close()


def sse_scaler_(k_rng , tfidf_matrix):
    sse_scaler  = []
    for k in k_rng:
        km = KMeans(n_clusters=k)
        km.fit(tfidf_matrix)
        km.predict(tfidf_matrix)
        sse_scaler.append(km.inertia_)
    #plot
    plt.plot(k_rng,sse_scaler)
    plt.xlabel("k")
    plt.ylabel("Sum of squared error")
    plt.show()
    plt.close()
