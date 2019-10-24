#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:53:19 2019

@author: amin
"""

#from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import normalized_mutual_info_score

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

print(__doc__)

# Generating the sample data from make_blobs
# This particular setting has one distinct cluster and 3 clusters placed close
# together.

range_n_clusters = 150

perm = np.zeros((10,2))
dat = pd.DataFrame({'NMI': [], 'Approach': []})

for i in range(0,10):
    subid1 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_DEC_no-sub-{:02d}_.txt'.format(i+1)
    X1 = pd.read_csv(subid1, header=None).values.flatten()
    print(len(np.unique(X1)))
    subid2 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_DEC_sub-{:02d}_notinf.txt'.format(i+1)
    Y1 = pd.read_csv(subid2, header=None).values.flatten()
    print(len(np.unique(Y1)))

    
    subid1 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_Kmeans_no-sub-{:02d}_.txt'.format(i+1)
    X2 = pd.read_csv(subid1, header=None).values.flatten()
    print(len(np.unique(X2)))
    subid2 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_Kmeans_sub-{:02d}_.txt'.format(i+1)
    Y2 = pd.read_csv(subid2, header=None).values.flatten()
    print(len(np.unique(Y2)))
    print([])
    
    perm[i,0] = normalized_mutual_info_score(X1, Y1, average_method='arithmetic')
    df = pd.DataFrame({'NMI': perm[i,0], 'Approach': ['DEC']}) 
    dat = dat.append(df)
    perm[i,1] = normalized_mutual_info_score(X2, Y2, average_method='arithmetic')
    df = pd.DataFrame({'NMI': perm[i,1], 'Approach': ['k-means']}) 
    dat = dat.append(df)

        

g = sns.catplot(x="Approach", y="NMI", kind="boxen", data=dat)
sns.swarmplot(x="Approach", y="NMI", color="k", size=3, data=dat, ax=g.ax);
#    rug_kws={'color': 'black'}
#    plt.xscale('log')

plt.suptitle(("Individuals' Clustering NMI\nDEC vs k-means"),
             fontsize=14, fontweight='bold')

plt.show()