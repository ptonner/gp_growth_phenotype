import matplotlib.pyplot as plt
import scipy, GPy, os, ast
import pandas as pd
import matplotlib as mpl

meta = data = parent = condition = None

"""
*************************
Plotting
*************************
"""

def plotSamples(samples,x=None,color='b',colors=None,plotMethod=None,*args,**kwargs):
	if x is None:
		x = range(samples.shape[0])
	if colors is None:
		colors = [color]*samples.shape[1]
	if plotMethod is None:
		plotMethod = plt.plot

	for i,c in zip(range(samples.shape[1]),colors):
		plotMethod(x,samples[:,i],color=c,*args,**kwargs)

"""
*************************
Data
*************************
"""

def setGlobals(data=None,meta=None,parent=None,condition=None):
	if not data is None:
		global data
		data = data

	if not meta is None:
		global meta
		meta = meta

	if not parent is None:
		global parent
		parent = parent

	if not condition is None:
		global condition
		condition = condition

def tidyfy(pivot):
	return pd.melt(pivot,id_vars=meta.columns.tolist(),value_vars=data.index.tolist(),
		value_name='OD',var_name='time')

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
	if xnull.shape[1] == 1:
		xnull = xnull[:,None]
	#if 'strain' in dims:
	#    xnull.strain = (xnull.strain!=parent).astype(int)
	xnull = xnull.values

	y = tidy.OD.values[:,None]
	k = GPy.kern.RBF(xnull.shape[1])
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
	results = results

	for i,m in enumerate(mutants):

		if (i + 1)%10 == 0:
			print 1.*i/len(mutants),m

		select = ((meta.Condition==control) | (meta.Condition==condition)) & (meta.strain.isin([parent,m]))
		if m in resultsParaquat:
			if len(resultsParaquat[m][1]) < numPerm:
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
