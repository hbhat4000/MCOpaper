import scipy.io as io
import numpy as np
import pandas as pd

rmse = np.zeros(2)
for i,j in zip([40,60],[0,1]):
    r = io.loadmat('%dtrain_naive_new2'%i)
    s = io.loadmat('%dtrain_fixed_new2'%i)

    model_n = r['model']
    model_n = np.concatenate(model_n)
    model_n = np.hstack(model_n)
    model_n = np.ravel(model_n)

    actual_n = r['actual']
    actual_n = np.concatenate(actual_n)
    actual_n = np.hstack(actual_n)
    actual_n = np.ravel(actual_n)

    model_f = s['model']
    model_f = np.concatenate(model_f)
    model_f = np.hstack(model_f)
    model_f = np.ravel(model_f)

    actual_f = s['actual']
    actual_f = np.concatenate(actual_f)
    actual_f = np.hstack(actual_f)
    actual_f = np.ravel(actual_f)

    # RMSE for naive and fixed models
    naive = np.sqrt(np.mean((model_n - actual_n)**2))
    fixed = np.sqrt(np.mean((model_f - actual_f)**2))

    # See improvment
    rmse[j]=np.abs(fixed-naive)/naive

ind = [40,60]
dat = {'RMSE':rmse}
df = pd.DataFrame(data=dat,index=ind)
df = df.rename_axis('Training Games')
print(df.to_latex(column_format = 'c'))
