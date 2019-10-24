#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:56:28 2019

@author: amin
"""

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

randperm = np.zeros((10,1000))
perm = np.zeros((10,1))

fig, axes = plt.subplots(4, 5)
#, sharex='col', sharey='row'
fig.set_size_inches(12, 7)
fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
for i in range(0,10):
    print(i+1)
    subid1 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_DEC_no-sub-{:02d}_.txt'.format(i+1)
    X = pd.read_csv(subid1, header=None).values.flatten()
    subid2 = '/p/himmelbach/amin/9T_MNI/150Cluster_Labels_DEC_sub-{:02d}_notinf.txt'.format(i+1)
    Y = pd.read_csv(subid2, header=None).values.flatten()
    
    perm[i,0] = normalized_mutual_info_score(X, Y, average_method='arithmetic')
    
       
    for j in range(1000):
        Y1 = np.random.permutation(Y)
        randperm[i,j] += normalized_mutual_info_score(X, Y1, average_method='arithmetic')

#    bins=np.arange(randperm[i,:].min(), randperm[i,:].max()+0.00002, 0.00002)
    lowb = randperm[i,:].mean() - 3*randperm[i,:].std()
    highb = randperm[i,:].mean() + 3*randperm[i,:].std()
    stp = randperm[i,:].std()/60
    bins=np.arange(lowb, highb, stp)
    permhist, permedges = np.histogram(randperm[i,:],100)
    permhist=permhist[np.newaxis,:]
#    print(permhist)
#    print(randperm[i,:])
    extent=[randperm[i,:].min(), randperm[i,:].max(), 0, 1]
    
    orgval = str(round(perm[i,0],3))
    permean = str(round(randperm[i,:].mean(),4))
    permstd = str(round(randperm[i,:].std(),5))
    
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].imshow(permhist, aspect = "auto", cmap="CMRmap", extent=extent)
    sns.distplot(randperm[i,:], hist = True, bins=100,  kde = True, kde_kws = {'shade': True, 'linewidth': 2}, ax=axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))])
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].set_xlim(bins.min(), bins.max())
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].set_xlim(bins.min(), bins.max())
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].ticklabel_format(axis='x', style='sci', scilimits=(-4,-4))
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].set_yticklabels([])
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].tick_params(axis='y', which='both', length=0)
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].tick_params(axis='x', labelsize = 10)
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].ticklabel_format(axis='x', style='sci', scilimits=(-4,-4))
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].ticklabel_format(axis='y', style='sci', scilimits=(3,3))
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].tick_params(axis='x', labelsize = 10)
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].axvline(randperm[i,:].mean(), color='r', linestyle='-')
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].axvline(randperm[i,:].std(), color='g', linestyle='--')
#    rug_kws={'color': 'black'}
#    plt.xscale('log')

    
    Snum = "Subject {} ".format(i+1)
    axes[int(math.floor(i/5))*2, int(math.fmod(i, 5))].set_title(Snum)
    Snum = "Individual map NMI = {} ".format(orgval)
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].set_title(Snum, fontsize = 'small')
    
    axes[int(math.floor(i/5))*2+1, int(math.fmod(i, 5))].legend({'Mean = {}'.format(permean):randperm[i,:].mean(),'Std = {}'.format(permstd):randperm[i,:].std()}, fontsize = 'x-small', loc = 'upper right')
    
    plt.suptitle(("Individuals NMI vs. Random permutaion histogram\n"
                  "DEC with k = %d" % range_n_clusters),
                 fontsize=14, fontweight='bold')

plt.tight_layout()            
plt.show()