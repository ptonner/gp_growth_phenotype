import pandas as pd
import numpy as np
import scipy
import seaborn as sns
from scipy.optimize import curve_fit
from gp_growth import gompertz, factory, metric,plot, normal
from gp_growth.storage import mongo
import GPy
import matplotlib.pyplot as plt
import bgreat


data = pd.read_csv("data/example/data.csv",index_col=0)
meta = pd.read_csv("data/example/meta.csv")

meta.head()

meta.shape
data.shape
assert data.shape[1] == meta.shape[0]

parent = 'parent'
control = 'control'
condition = 'stress'

meta['strain-regression'] = (meta.strain!=parent).astype(int)
meta['condition'] = (meta.Condition!=control).astype(int)
meta['interaction'] = meta['strain-regression']*meta.condition

plt.figure(figsize=(16,6))
plt.suptitle("Before Formatting")
plt.subplot(121)
plt.title('control')

bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain==parent))[0]],color='k',label='parent')
bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
plt.ylim(0.05,.7)
plt.subplot(122)
plt.title('stress')
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)&
                    (meta.strain==parent))[0]],color='k',label='parent')
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)&
                    (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
plt.ylim(0.05,.7)

plt.show()