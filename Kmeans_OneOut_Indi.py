#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:40:32 2019

@author: amin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.


def Clust(sub, X):
    
       
    range_n_clusters = 150
    
#    subid1 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_Kmeans_no-sub-{:02d}_.txt'.format(sub)
#    
#    count = 0
#    Y = np.zeros((len(X), 144))
#
#    for j in range(len(X[1,:])):
#        if (int(math.fmod(j-sub+1,10)) != 0):
#            Y[:,count] = X[:,j]
#            count += 1
#            
#    clusterer = KMeans(n_clusters=range_n_clusters, random_state=20, n_init= 40, max_iter=500, tol=1e-6, n_jobs=-1)
#    y_pred1 = clusterer.fit_predict(Y)
#    
#    np.savetxt(subid1, y_pred1)
#    
    
    subid1 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_Kmeans_sub-{:02d}_.txt'.format(sub)
    
    count = 0
    Y = np.zeros((len(X), 16))

    for j in range(len(X[1,:])):
        if (int(math.fmod(j-sub+1,10)) == 0):
            Y[:,count] = X[:,j]
            count += 1
            print(count)
            
    clusterer = KMeans(n_clusters=range_n_clusters, random_state=20, n_init= 40, max_iter=500, tol=1e-6, n_jobs=-1)
    y_pred2 = clusterer.fit_predict(Y)
    
    np.savetxt(subid1, y_pred2)
    
    return y_pred2, #y_pred1





#with closing(Pool(10)) as p:
#    p.map(Clust, [x for x in range(1,11)])
#    p.terminate()
  
X = pd.read_csv('Data_MNI_EntireBox_noT1_spatiallyinformed_VoxelLoc_MainGrad_neworient.csv', header=None).values
    
for i in range(1,11):
    print(i)
    Clust(i, X)
    