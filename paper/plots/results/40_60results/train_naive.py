"""
This code plots test results for 40/60 training-test split
for fixed CTMC
"""

# Read necessary modules
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr

# Load data 40 and 60train_naive.mat
for i in [40,60]:
    r = io.loadmat('%dtrain_naive_new1'%i)
    model = r['model']
    model = np.concatenate(model)
    model = np.hstack(model)
    model = np.ravel(model)
    actual = r['actual']
    actual = np.concatenate(actual)
    actual = np.hstack(actual)
    actual = np.ravel(actual)

    rho, _ = pearsonr(model,actual)
    print(rho)
    model_max = np.max(model)
    actual_max = np.max(actual)
    val = max(model_max,actual_max)
    plt.scatter(model,actual,c='k')
    plt.plot([0.001,2*val],[0.001,2*val],'r')
    plt.axis([0.001,2*val,0.001,2*val])
    plt.xscale('log',basex=10)
    plt.yscale('log',basey=10)
    plt.xlabel('simulated playing time',fontsize=14)
    plt.ylabel('true playing time',fontsize=14)
    plt.title('Test results -- %d training games'%i,fontsize=16)
    plt.savefig('%dtrain_naive_new1.eps'%i,format='eps',dpi=300)

