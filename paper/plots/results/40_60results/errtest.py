import scipy.io as io
import numpy as np

r = io.loadmat('60train_naive')
s = io.loadmat('60train_fixed')

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
naive = np.sqrt(np.sum((model_n - actual_n)**2))
fixed = np.sqrt(np.sum((model_f - actual_f)**2))

# See improvment
print((fixed-naive)/naive)

