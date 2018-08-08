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
#teams = ['GS','Bkn']
fig = plt.figure()
ax = plt.subplot(111)
colors = cm.rainbow(np.linspace(0,1,len(teams)))

l = 0
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
    numtrain = 40
    for i in tokeep[:numtrain]:
        t = homeoraway(i)
        tg = dub[dub.rawdate == i]
        tt = np.array(tg.timePlayed)
        #new = np.append(new,np.cumsum(time))
        new = np.cumsum(tt)
        if (t == 1):
            lineup = np.array(tg.HLu,dtype = 'int')-1
        else:    
            lineup = np.array(tg.VLu,dtype = 'int')-1
        for j in range(len(lineup)):
            alltimes[lineup[j]] += tt[j]
        nrs = np.where(np.diff(lineup)!=0)[0]
        maxlu.append(np.max(lineup))
        fixedLU = lineup[nrs]
        fixedtime = new[nrs]
        localcounts += counttrans(fixedLU,w)
        newtimes += statetimetally(fixedLU,fixedtime,w)
    
    maxstate = np.max(maxlu)
    #choice = ['Naive','fixed']
    choice = ['Naive']
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
    for i in choice:
        if (i == 'Naive'):
            phat = createCTMC(thismat,newtimes,ns)
            #print(phat)
        else:
            transmat = createCTMC(thismat,newtimes,ns)
            eps, constrviol = fixCTMC(transmat,statefrac, forcePos = True)
            l = ns*(ns-1)
            phat = addPert(transmat, eps[0:l], ns, 'CTMC')
            #print(phat)
        pivec = equilib(phat,'CTMC')
        #LTerr[k] = np.sum(np.abs(pivec - statefrac)) 
        pivector.append(pivec)
        k += 1
        #print(LTerr)
    
    #p = np.zeros(len(realtime))
    #p[goodstates] = pivec 
    #traintime = time.time() - start
    
    test_times = np.zeros(w) 
    #fixed_pivec = []
    #testdat = dub[dub.rawdate >= tokeep[numtrain]]
    
    
    maxtest = []
    for i in tokeep[numtrain:]:
        t = homeoraway(i)
        tg = dub[dub.rawdate == i]
        tt = np.array(tg.timePlayed)
        #new = np.append(new,np.cumsum(time))
        new = np.cumsum(tt)
        if (t == 1):
            lineup = np.array(tg.HLu,dtype = 'int')-1
        else:    
            lineup = np.array(tg.VLu,dtype = 'int')-1
        for j in range(len(lineup)):
            alltimes[lineup[j]] += tt[j]
        nrs = np.where(np.diff(lineup)!=0)[0]
        maxtest.append(np.max(lineup))
        fixedLU = np.append(lineup[nrs],lineup[-1])
        fixedtime = np.append(new[nrs],new[-1])
        test_times += statetimetally(fixedLU,fixedtime,w)
        test_times[fixedLU[0]] += new[0]

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
    len_choice = len(choice)
    #for i in range(len_choice):
        #models.append(fixed_pivec[i]*n)
        #alltt.append(test_times)
    allnaive = fixed_pivec[0]*n
    realt_naive = test_times
    model.append(allnaive)
    actual.append(realt_naive)
    #alltt.append(test_times)
    #allttmodel.append(pivec*n)
    #print(np.sum(np.abs(realtime-pivec*n)))
    #for u, c in zip(allnaive,colors):
    ax.scatter(realt_naive, allnaive,color=colors[l], label='%s'%r)
    l += 1

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
ax.set_xlabel('real time (thousands of seconds)')
ax.set_ylabel('model time (thousands of seconds)')
ax.set_title('NBA test (naive) results')
ax.legend(loc='center left', ncol=2, bbox_to_anchor=(1,0.5),prop={'size':7})
plt.savefig('naive_%dgames_test.eps'%numtrain)
plt.savefig('naive_%dgames_test.pdf'%numtrain)
#realtime = np.concatenate(totaltt)
#modeltime = np.concatenate(totalmodels)
dat = {'model': model, 'actual': actual}
io.savemat('%dtrain_naive'%numtrain,dat)
    
"""
plt.scatter(realt_naive,allnaive)
plt.plot(realt_naive,realt_naive,'k')
plt.xlim(-0.1,np.round(np.max(realt_naive)))
plt.ylim(-1,np.round(np.max(allnaive)))
plt.xlabel('real time (thousands of seconds)')
plt.ylabel('model time (thousands of seconds)')
plt.legend(['$y = x$','model (naive)'])
plt.title('Test time results for %s' %r)
plt.savefig('testresult_naive_%s.eps' %r)
"""

"""
traintime = time.time() - start
print(np.correlate(realtime,nntimes))
plt.scatter(realtime,nntimes)
plt.savefig('NBAsimvsreal.eps') 
#print(np.sum(np.abs(realtime - nntimes)))
#print(training_time)
#print(LTerr)
"""
"""
    test_times = np.zeros(w)
    whole_times = np.zeros(w)
    whole_times[goodstates] = newtimes
    for i in tokeep[N:]:
        x = homeoraway(i)
        tg = dub[dub.rawdate == i]
        tt = np.array(np.cumsum(tg.timePlayed))
        if (x == 1):
            states = np.array(tg.HLu,dtype = 'int') - 1
        else:
            states = np.array(tg.VLu,dtype = 'int') - 1 
        maxlu.append(np.max(states))
        tofix = np.where(np.diff(states)!=0)[0]
        test_times += statetimetally(states[tofix],tt[tofix],w)
        whole_times += statetimetally(states[tofix],tt[tofix],w)

    w1 = np.max(maxlu)
    fixed_pivec = []
    if w < w1:
        test_times = np.zeros(w1)
        whole_times = np.zeros(w1)
        whole_times[goodstates] = newtimes[goodstates]
        for i in pivector:
            p = np.zeros(w1)
            p[goodstates] = i[goodstates]
            fixed_pivec.append(p)
    else:
        for i in pivector:
            p = np.zeros(w)
            p[goodstates] = i
            fixed_pivec.append(p)

    statefrac_test = test_times/np.sum(test_times)
    whole_test = whole_times/np.sum(whole_times)
    #print(statefrac_test)
    #print(whole_test)
    test_err = np.zeros(2)
    totalerr = np.zeros(2)
    for i in range(2): 
        test_err[i] = np.sum(np.abs(statefrac_test - fixed_pivec[i]))
        totalerr[i] = np.sum(np.abs(whole_test - fixed_pivec[i]))


    err[m,0] = LTerr[0]
    err[m,1] = LTerr[1]
    err[m,2] = test_err[0]
    err[m,3] = test_err[1]
    err[m,4] = totalerr[0]
    err[m,5] = totalerr[1]
    m += 1

err_dat = pd.DataFrame(err, columns = ['LT Training Naive','LT Training Fixed','LT Test Naive','LT Test Fixed','LT Total Naive', 'LT Total Fixed'],index = teams)
err_dat = err_dat.rename_axis('1-norm Error',axis = 1).rename_axis('Team')
print(err_dat.to_latex(column_format = 'lcccccc'))
"""
