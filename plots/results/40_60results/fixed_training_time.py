# This time is for real. 

# We are using the NBA data for the season 2015-16

import numpy as np
import pandas as pd
from mco1 import *
import time 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.io as io
# Read team csv file

teams = ['Atl','Bkn','Bos','Cha','Chi','Cle','Dal','Den','Det','GS','Hou',
'Ind','LAC','LAL','Mem','Mia','Mil','Min','NO','NY','OKC','Orl','Phi','Pho',
'Por','SA','Sac','Tor','Uta','Was']

def statetimetally(ts,times,N):
    stally = np.zeros(N)
    diffseq = np.diff(times)
    for i in range(len(ts)-1):
        stally[ts[i+1]] += diffseq[i]
    stally[ts[0]] += times[0]
    return stally

#teams = ['GS','Cle']
#fig = plt.figure()
#ax = plt.subplot(111)
#colors = cm.rainbow(np.linspace(0,1,len(teams)))
sparsity = np.zeros(30)
LTerr = np.zeros(30)
start = time.time()
for r in teams:
    dub = pd.read_csv(str(r)+'.csv')
    dub = dub.drop('Unnamed: 0', axis = 1)
    numrows = dub.shape[0]
    w = np.max([np.array(dub.HLu),np.array(dub.VLu)])
    dates = np.unique(dub.rawdate)
    # First figure out dates we don't want.
    toskip = []
    for i in dates:
        tt = dub[dub.rawdate == i].timePlayed
        if np.max(np.cumsum(tt)) > 2880:
            toskip.append(i)
    # These are the dates to use
    tokeep = [i for i in dates if i not in toskip]

    start = time.time()
    def homeoraway(date):
        test = np.array(dub[dub.rawdate == date].Hteam)
        if (test[0] == r):
            x = 1
        else:
            x = 0
        return x

    localcounts = np.zeros((w,w),dtype = 'int')
    newtimes = np.zeros(w)
    maxlu = []
    alltimes = np.zeros(w)
    numtrain = 60
    for i in tokeep[:numtrain]:
        t = homeoraway(i)
        tg = dub[dub.rawdate == i]
        tt = tg.timePlayed.values
        #new = np.append(new,np.cumsum(time))
        new = np.cumsum(tt)
        if (t == 1):
            lineup = np.array(tg.HLu,dtype = 'int')-1
        else:    
            lineup = np.array(tg.VLu,dtype = 'int')-1
        for j in range(len(lineup)):
            alltimes[lineup[j]] += tt[j]
        nrs = np.where(np.diff(lineup)!=0)[0]
        #nrs = np.append(nrs,len(lineup)-1)
        maxlu.append(np.max(lineup))
        #fixedLU = lineup[nrs]
        #fixedtime = new[nrs]
        fixedLU = np.append(lineup[nrs],lineup[-1])
        fixedtime = np.append(new[nrs],new[-1])
        localcounts += counttrans(fixedLU,w)
        newtimes += statetimetally(fixedLU,fixedtime,w)
        #newtimes[fixedLU[0]] += fixedtime[0]
      
    maxstate = np.max(maxlu)
    #choice = ['Naive','fixed']
    #choice = ['fixed']
    #statefrac = newtimes/np.sum(newtimes)
    #pivector = []
    #LTerr = np.zeros(2)
    ns = maxstate + 1
    realtime = newtimes[:ns]
    #realtime = alltimes[:ns]
    statefrac = realtime/np.sum(realtime)
    thismat = localcounts[0:ns,:]
    thismat = localcounts[:,0:ns]
    pivector = []
    k = 0
    np.random.seed(7)
    transmat = createCTMC(thismat,newtimes,ns)
    f = np.array([0.01,0.1,10,100,200,300])
    for m in f:
        eps, constrviol = fixCTMC(transmat,statefrac,m,forcePos = True)
        #eps_flat = np.ravel(eps)
        #sparsity[o] = np.sum(np.ravel(eps)!=0.)/len(np.ravel(eps))
        l = ns*(ns-1)
        phat = addPert(transmat, eps[0:l], ns, 'CTMC')
        #print(phat)
        pivec = equilib(phat,'CTMC')
        if (np.sum(np.abs(pivec - statefrac)) > 1e-07):
            continue
        else:
            break
    pivec
    sparsity[k] = np.sum(np.ravel(eps)!=0.)/len(np.ravel(eps))
    LTerr[k] = np.sum(np.abs(pivec - statefrac)) 
    pivector.append(pivec)
    k += 1
    #print(LTerr)
    
    #p = np.zeros(len(realtime))
    #p[goodstates] = pivec 
traintime = time.time() - start
print(traintime) 
