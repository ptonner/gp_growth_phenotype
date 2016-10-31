import matplotlib.pyplot as plt
import numpy as np
import scipy, GPy, os, ast
import pandas as pd
import matplotlib as mpl

meta = data = parent = condition = control = None

"""
*************************
Plotting
*************************
"""

def plotSamples(samples,x=None,color='b',colors=None,plotMethod=None,label=None,*args,**kwargs):
	if x is None:
		x = range(samples.shape[0])
	if colors is None:
		colors = [color]*samples.shape[1]
	if plotMethod is None:
		plotMethod = plt.plot

	for i,c in zip(range(samples.shape[1]),colors):
		if not label is None:
			plotMethod(x,samples[:,i],color=c,label=label,*args,**kwargs)
			label = None
		else:
			plotMethod(x,samples[:,i],color=c,*args,**kwargs)

"""
*************************
Data
*************************
"""

def setGlobals(_data=None,_meta=None,_parent=None,_condition=None,_control=None):
	if not _data is None:
		global data
		data = _data

		data.columns = range(data.shape[1])

	if not _meta is None:
		global meta
		meta = _meta

	if not _parent is None:
		global parent
		parent = _parent

	if not _condition is None:
		global condition
		condition = _condition

	if not _control is None:
		global control
		control = _control

def getData(select,fmt='standard'):
	temp = data.loc[:,meta.index[select]]
	temp2 = meta.loc[select,:]

	if fmt == 'standard':
		return temp,temp2
	pivot = pd.concat((temp2,temp.T),1)
	if fmt == 'pivot':
		return pivot

	tidy = pd.melt(pivot,id_vars=meta.columns.tolist(),value_vars=data.index.tolist(),
		value_name='OD',var_name='time')

	return tidy

def tidyfy(pivot):
	return pd.melt(pivot,id_vars=meta.columns.tolist(),value_vars=data.index.tolist(),
		value_name='OD',var_name='time')

def pivotify(temp,temp2):
	return pd.concat((temp2,temp.T),1)

def buildTable(results):

	ret = pd.DataFrame()

	for k in results.keys():
		row = pd.Series(results[k],name=k,index=['actual','permuted'])
		ret = pd.concat((ret,row),1,ignore_index=False)

	ret = ret.T
	ret['FDR'] = ret.apply(lambda x: 1.*sum(x['actual']<=x['permuted'])/len(x['permuted']),1)
	ret = ret.sort_index(by='FDR')

	return ret

"""
*************************
Analysis
*************************
"""

def permTest(pivot,nullLoglikelihood,numPerm=10,permCol='strain-regression',dims=[],colTransform={},timeThin=3,timeSelect=None):
	pivot = pivot.copy()
	perms = []

	for i in range(numPerm):
		pivot[permCol] = np.random.choice(pivot[permCol],size=pivot.shape[0],replace=False)
		tidy = tidyfy(pivot)

		#timeSelect = tidy.time.unique()[::timeThin]
		if not timeSelect is None:
			tidy = tidy[tidy.time.isin(timeSelect)]

		#_dims = ['time','strain-regression']+dims
		_dims = dims

		x = tidy[_dims]
		#x.strain = (x.strain!=parent).astype(int)
		for k in colTransform.keys():
			x[k] = colTransform[k](x)
		x = x.values

		y = tidy.OD.values[:,None]
		k = GPy.kern.RBF(x.shape[1],ARD=True)

		gp = GPy.models.GPRegression(x,y,k)
		gp.optimize()
		perms.append(gp.log_likelihood()-nullLoglikelihood)

	return perms

def selectStrain(m):
	return ((meta.Condition==control) | (meta.Condition==condition)) & (meta.strain.isin([parent,m]))

def buildGP(select,timeThin=4,dims=[]):

	temp = data.loc[:,meta.index[select]]
	temp2 = meta.loc[select,:]

	pivot = pivotify(temp,temp2) #pd.concat((temp2,temp.T),1)
	tidy = tidyfy(pivot)

	timeSelect = tidy.time.unique()[::timeThin]
	tidy = tidy[tidy.time.isin(timeSelect)]

	dims = ['time','strain-regression']+dims

	x = tidy[dims]
	x = x.values
	y = tidy.OD.values[:,None]
	k = GPy.kern.RBF(x.shape[1],ARD=True)
	gp = GPy.models.GPRegression(x,y,k)
	gp.optimize()
	return gp

def runTest(select,numPerm=10,timeThin=3,dims=[],nullDim='strain-regression',colTransform={}):
	#select = (meta.Condition==control) & (meta.strain.isin([parent,m]))

	temp = data.loc[:,meta.index[select]]
	temp2 = meta.loc[select,:]
	pivot = pd.concat((temp2,temp.T),1)

	#tidy = pd.melt(pivot,id_vars=meta.columns.tolist(),value_vars=data.index.tolist(),
	#    value_name='OD',var_name='time')
	tidy = tidyfy(pivot)

	timeSelect = tidy.time.unique()[::timeThin]
	tidy = tidy[tidy.time.isin(timeSelect)]

	dims = ['time','strain-regression']+dims

	x = tidy[dims]
	# x.strain = (x.strain!=parent).astype(int)
	x = x.values
	y = tidy.OD.values[:,None]
	k = GPy.kern.RBF(x.shape[1],ARD=True)

	gp = GPy.models.GPRegression(x,y,k)
	gp.optimize()

	altLoglikelihood = gp.log_likelihood()

	_dims = [d for d in dims]
	_dims.remove(nullDim)

	xnull = tidy[_dims]
	xnull = xnull.values
	if xnull.ndim == 1:
		xnull = xnull[:,None]

	y = tidy.OD.values[:,None]
	k = GPy.kern.RBF(xnull.shape[1],ARD=True)
	gp = GPy.models.GPRegression(xnull,y,k)
	gp.optimize()

	nullLoglikelihood = gp.log_likelihood()
	actualBf = altLoglikelihood - nullLoglikelihood

	perms = permTest(pivot,nullLoglikelihood,
				#permCol='strain-regression',numPerm=numPerm,
				permCol=nullDim,numPerm=numPerm,
				dims=dims,colTransform=colTransform,
				timeSelect=timeSelect)

	return actualBf, perms

def testMutants(mutants,numPerm=10,timeThin=4,dims=[],nullDim='strain-regression'):
	results = {}

	for i,m in enumerate(mutants):

		if (i + 1)%10 == 0:
			print 1.*i/len(mutants),m

		select = ((meta.Condition==control) | (meta.Condition==condition)) & (meta.strain.isin([parent,m]))
		if m in results:
			if len(results[m][1]) < numPerm:
				results[m][1].extend(runTest(select,timeThin=timeThin,
					dims=dims,nullDim=nullDim,
					numPerm=numPerm-len(results[m][1]))[1])
			continue

		results[m] = runTest(select,timeThin=timeThin,
			dims=dims,nullDim=nullDim,
			numPerm=numPerm)

	return buildTable(results)

def testMutantControl(mutants,**kwargs):
	return testMutants(mutants,**kwargs)

def testMutantCondition(mutants,**kwargs):
	return testMutants(mutants,dims=['condition','interaction'],nullDim='interaction',**kwargs)
