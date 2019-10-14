import matplotlib.pyplot as plt
import numpy as np
import bgreat



import pandas as pd
data = pd.read_csv("data/example/data.csv",index_col=0)
meta = pd.read_csv("data/example/meta.csv")

data.head()

meta.head()

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
bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain==parent))[0]],color='k',label=parent)
bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
# plt.ylim(0.05,.7)

plt.subplot(122)
plt.title('stress')
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)& (meta.strain==parent))[0]],color='k',label=parent)
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)& (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
plt.show()



data = data.iloc[4:,:]
data = np.log2(data)
g = meta.groupby(['strain','Condition'])
for k,ind in enumerate(g.groups):
    print(k)
    print(ind)
    data.iloc[:,g.groups[ind]] -= data.iloc[0,g.groups[ind]].mean()



plt.figure(figsize=(16,6))
plt.suptitle("After Formatting")

plt.subplot(121)
plt.title('control')
bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain==parent))[0]],color='k',label=parent)
bgreat.plotSamples(data.values[:,np.where((meta.condition==0)& (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
plt.ylim(-.4,2.4)

plt.subplot(122)
plt.title('stress')
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)& (meta.strain==parent))[0]],color='k',label='parent')
bgreat.plotSamples(data.values[:,np.where((meta.condition==1)& (meta.strain!=parent))[0]],color='g',label='mutant')
plt.legend(loc='best')
plt.ylim(-.4,2.4)
plt.show()




##import bgreat
##bgreat.setGlobals(_data=data, _meta=meta)
##bgreat.setGlobals(_parent=parent,_control=control)

bgreat.setGlobals(_data=data,_meta=meta,_parent=parent,_control=control,_condition=None)

mutants = ['mutant']

results = bgreat.testMutantControl(mutants,numPerm=20,dims=['time','strain-regression'])

results

plt.hist(results.permuted.values[0])
plt.show()

gp = bgreat.buildGP(bgreat.selectStrain('mutant'))

gp

xpred = np.zeros((100,2))
xpred[:50,0] = np.linspace(data.index.min(),data.index.max())
xpred[50:,0] = np.linspace(data.index.min(),data.index.max())

xpred[50:,1] = 1

mu,cov = gp.predict(xpred,full_cov=True)
var = np.diag(cov)
mu = mu[:,0]

plt.figure(figsize=(10,5))

plt.plot(xpred[:50,0],mu[:50],label='parent');
plt.fill_between(xpred[:50,0],mu[:50]-2*np.sqrt(var[:50]),mu[:50]+2*np.sqrt(var[:50]),alpha=.1)

plt.plot(xpred[:50,0],mu[50:],label='mutant')
plt.fill_between(xpred[:50,0],mu[50:]-2*np.sqrt(var[50:]),mu[50:]+2*np.sqrt(var[50:]),alpha=.1)

plt.legend(fontsize=20)
plt.show()