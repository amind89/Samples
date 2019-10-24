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

X = pd.read_csv('Data_basic_sym_no-height_neworient.csv', header=None).as_matrix()
#X = X[:10000,:]

range_n_clusters = 120


# Create a subplot with 1 row and 2 columns
fig, axes = plt.subplots(5, 7)
fig.set_size_inches(12, 7)
fig.subplots_adjust(hspace=0.4, wspace=0.2)


clusterer = KMeans(n_clusters=range_n_clusters, random_state=20, n_init= 40, max_iter=500, tol=1e-6, n_jobs=-1)
cluster_labels = clusterer.fit_predict(X)

subid2 = '/p/himmelbach/amin/9T_MNI/120Cluster_Labels_basic_sym_no-height_neworient.txt'
np.savetxt(subid2, cluster_labels)


b = 111*111

for i in range(34):
    a = cluster_labels[i*b:(i+1)*b,]
    a = a.reshape(111, 111)
    a = np.rot90(a, 1)
      
    axes[int(4-math.floor(i/7)), 6-int(math.fmod(i, 7))].imshow(a, cmap='Paired')
    
    Snum = "slice {}".format(34-i)
    axes[int(4-math.floor(i/7)), 6-int(math.fmod(i, 7))].set_title(Snum)
    


plt.suptitle(("KMeans clustering on midbrain data "
              "with range_n_clusters = %d" % range_n_clusters),
             fontsize=14, fontweight='bold')

plt.show()