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

model = []
actual = []
# Calculate these two quantities via average.
esparse = np.zeros(30)
viol = np.zeros(30)
N = np.zeros(30)
numeps = np.zeros(30)
#teams = ['GS','Cle']
fig = plt.figure()
ax = plt.subplot(111)
colors = cm.rainbow(np.linspace(0,1,len(teams)))
#sparsity = np.zeros(30)
u = 0
o = 0
#start = time.time()
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
        test = dub[dub.rawdate == date].Hteam.values[0]
        if (test == r):
            x = 1
        else:
            x = 0
        return x
    
    localcounts = np.zeros((w,w),dtype = 'int')
    newtimes = np.zeros(w)
    maxlu = []
    alltimes = np.zeros(w)
    numtrain = 40
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
        newtimes += statetimetally1(fixedLU,fixedtime,w)
        #newtimes[fixedLU[0]] += fixedtime[0]
      
    maxstate = np.max(maxlu)
    #choice = ['Naive','fixed']
    choice = ['fixed']
    #statefrac = newtimes/np.sum(newtimes)
    #pivector = []
    #LTerr = np.zeros(2)
    ns = maxstate + 1
    
    realtime = newtimes[:ns]
    #realtime = alltimes[:ns]
    statefrac = realtime/np.sum(realtime)
    thismat = localcounts[0:ns,:]
    thismat = thismat[:,0:ns]
    pivector = []
    k = 0
    
    np.random.seed(7)
    for i in choice:
        if (i == 'Naive'):
            phat = createCTMC(thismat,newtimes,ns)
            #print(phat)
        else:
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
                    esparse[o] += np.sum(np.ravel(eps[0:l])!=0.)
                    viol[o] += constrviol
                    numeps[o] += l
                    N[o] += ns
                    o += 1
                    break
            pivec
        #LTerr[k] = np.sum(np.abs(pivec - statefrac)) 
        pivector.append(pivec)
        k += 1
        #print(LTerr)
    
    #p = np.zeros(len(realtime))
    #p[goodstates] = pivec 
    traintime = time.time() - start
    #print(traintime) 
    test_times = np.zeros(w) 
    #fixed_pivec = []
    #testdat = dub[dub.rawdate >= tokeep[numtrain]]
    
    
    maxtest = []
    for i in tokeep[numtrain:]:
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
        nrs1 = np.where(np.diff(lineup)!=0)[0]
        
        #nrs1 = np.append(nrs1,len(lineup)-1)
        maxtest.append(np.max(lineup))
        fixedLU = np.append(lineup[nrs1],lineup[-1])
        fixedtime = np.append(new[nrs1],new[-1])
        test_times += statetimetally1(fixedLU,fixedtime,w)
        #test_times[fixedLU[0]] += fixedtime[0]

    m_test = np.max(maxtest)
    fixed_pivec = []
    if maxstate < m_test:
        test_times = test_times[:m_test+1]
        p = np.zeros(m_test+1)
         
        for i in pivector:
            p[:ns] = i 
            fixed_pivec.append(p)
    else:
        test_times = test_times[:ns] 
        for i in pivector:
            fixed_pivec.append(i)        
        
           
    m = len(tokeep[numtrain:])
    n = m*2880
    #len_choice = len(choice)
    #for i in range(len_choice):
        #models.append(fixed_pivec[i]*n)
        #alltt.append(test_times)
    allnaive = fixed_pivec[0]*n/1000
    realt_naive = test_times/1000
    model.append(allnaive)
    actual.append(realt_naive)
    #for u, c in zip(allnaive,colors):
    ax.scatter(realt_naive, allnaive, label='%s'%r)
    u += 1

mat = np.zeros((len(teams),4))
mat[:,0] = N
mat[:,1] = numeps
mat[:,2] = esparse
mat[:,3] = viol
result = pd.DataFrame(mat,index=teams,columns=['Matrix Row Number','# of epsilons','Nonzero eps','constraint #']) 
result = result.rename_axis('Team')
print(result.to_latex(column_format = 'lcccc'))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.set_xlabel('real time (thousands of seconds)')
ax.set_ylabel('model time (thousands of seconds)')
ax.set_title('NBA test (%s) results'%choice[0])
ax.legend(loc='center left', ncol=2, bbox_to_anchor=(1,0.5),prop={'size':7})
plt.savefig('%s_%dgames_test_new.eps'%(choice[0],numtrain),format='eps',dpi=300)
plt.savefig('%s_%dgames_test_new.pdf'%(choice[0],numtrain))
#print(sparsity)
#print(traintime)
#print(viol)
#print(esparse)
dat = {}
dat['model']= model 
dat['actual'] = actual
io.savemat('%dtrain_fixed_new2'%numtrain,dat)

    
