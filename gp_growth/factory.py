# Author: Peter Tonner (peter.tonner@duke.edu)

import GPy
import numpy as np
import pandas as pd
from categorical import Categorical

class Factory(object):
	"""Abstract factory class for defining how to contruct a GP.

	A factory takes input data in regression format, and generates a GP as specified by its input dimension, their corresponding kernel function, and the output dimension.
	Subclasses make calls to addInputDimension in the contructor to make customized factories for specific analyses.

	Attributes:
		model: GP contruction method (default: GP.models.GPRegression)
		inputDimensions: list of tuples of the form (n,k) where n is the name of the dimension and k is the associated kernel funcition
		outputDimension: output dimension name for GP
		savedParameters: pandas DataFrame for storing optimized hyperparameters from a GP, the columns of the dataframe are determined by the kernel built by the factory (see buildKernel)
	"""

	def __init__(self,normalize=False,model=None,**kwargs):
		"""Build a GP factory.

		Base contructor method. By default, time is assumed as an intput dimenison with an RBF kernel, and GPy.models.GPRegression is the GP class.

		Args:
			normalize: should data be normalized (mean and variance scaled) before input to GP
			model: specify a model for GP construction (default: GPy.models.GPRegression)
			kwargs: passed to constructor of GP model
		"""
		self.inputDimensions = []
		self.addInputDimension("time",GPy.kern.RBF)
		self.outputDimension = "od"

		self.normalize = normalize
		self.means = {}
		self.std = {}

		if model is None:
			self.model = GPy.models.GPRegression
		else:
			self.model = model

		self.model_kwargs = kwargs

		# storage for optimized parameters
		self.savedParameters = pd.DataFrame()

		# keeping records of categorical data seen
		# for usage see convertCategoricalDimension
		self.seenCategories = {}

	def addInputDimension(self,name,kernelType=None,**kwargs):
		if kernelType is None:
			kernelType = GPy.kern.RBF
			
		if not issubclass(kernelType,GPy.kern.Kern):
			raise ValueError("kernelType is not a valid kernel")

		self.inputDimensions.append((name,kernelType,kwargs))

	def inputColumns(self):
		return [x[0] for x in self.inputDimensions]

	def buildKernel(self):

		# columns to use for x input to GP
		input_columns = self.inputColumns()

		# build the kernels from input dimensions
		name,kern,kw = self.inputDimensions[0]
		ind = input_columns.index(name)
		kernel = kern(1,active_dims=[ind],name = name,**kw)

		for name,kern,kw in self.inputDimensions[1:]:
			ind = input_columns.index(name)
			kernel = kernel + kern(1,active_dims=[ind],name = name,**kw)

		return kernel

	def convertCategoricalDimension(self,x):
		"""Change a categorical variable into integer values, values seen previously will return the same value each time"""

		name = x.name
		vals = x.unique()

		if not name in self.seenCategories:
			self.seenCategories[name] = {}
		for v in vals:
			if not v in self.seenCategories[name]:
				self.seenCategories[name][v] = len(self.seenCategories[name].keys())

		temp = np.zeros(x.shape) - 1
		for i in range(temp.shape[0]):
			temp[i] = self.seenCategories[name][x.values[i]]

		assert all(temp!=-1),"error converting a category"

		return temp

		# temp = pd.get_dummies(x)
		# return np.where(temp==1)[1]


	def buildInputFixed(self,size=None,convert=True,**kwargs):
		""" build the data by using a single dynamic input (e.g. time) and many other fixed inputs (e.g. batch, strain...)
		"""

		if size is None:
			size = 100

		fixedParams = {}
		inputColumns = self.inputColumns()
		dynamicDimension = ""
		dynamicMin = dynamicMax = 0

		for k in kwargs.keys():
			if k in inputColumns:
				fixedParams[k] = kwargs[k]
			elif k.rstrip("_max") in inputColumns:
				dynamicDimension = k.rstrip("_max")
				dynamicMax = kwargs[k]
			elif k.rstrip("_min") in inputColumns:
				dynamicMin = kwargs[k]

		if len(fixedParams.keys()) < len(inputColumns) - 1:
			raise ValueError("not enough fixed inputs!")
		elif len(fixedParams.keys()) >= len(inputColumns):
			raise ValueError("too many fixed inputs!")

		x = np.zeros((size,len(inputColumns)),dtype=object)
		x[:,0] = np.linspace(dynamicMin,dynamicMax,size)
		tempColumns = [dynamicDimension]

		for i,k in enumerate(fixedParams.keys()):
			x[:,i+1] = fixedParams[k]
			tempColumns.append(k)

		x = pd.DataFrame(x,columns = tempColumns)

		if convert:
			return self.buildInput(x,renormalize=False)
		else:
			return x

	def buildInput(self,data=None,renormalize=True):
		x = data[self.inputColumns()].copy()

		for name,kern,kw in self.inputDimensions:
			if kern == Categorical:
				x[name] = self.convertCategoricalDimension(x[name])
			elif self.normalize:
				if renormalize:
					self.means[name] = x[name].mean()
					self.std[name] = x[name].std()
				x[name] = (x[name] - self.means[name])/self.std[name]

		# if self.normalize:
		# 	self.means = x.mean()
		# 	self.std = x.std()
		# 	x = (x - self.means)/x.std()

		x = x.values.astype(float)

		return x

	def buildOutput(self,data, renormalize=True):

		# if renormalize:
		# 	ret = data[self.outputDimension]
		# 	self.means[self.outputDimension] = ret.mean()
		# 	self.std[self.outputDimension] = ret.std()
		# 	ret = (ret - ret.mean())/ret.std()
		# 	return ret.values[:,np.newaxis]
		return data[self.outputDimension].values[:,np.newaxis]

	def build(self,data,optimize=False,useSaved=False,renormalize=True,**kwargs):

		# if "renormalize" in kwargs:
		# 	renormalize = kwargs['renormalize']
		# 	del kwargs['renormalize']
		# else:
		# 	renormalize=True

		x = self.buildInput(data,renormalize=renormalize)
		y = self.buildOutput(data,renormalize=renormalize)

		kernel = self.buildKernel()

		# get previously optimized hyperparameters if they exist
		if (not optimize) and useSaved: #(not name is None) and (name in self.savedParameters.index):
			select = [True] * self.savedParameters.shape[0]
			for k,v in kwargs.iteritems():
				select = np.all((select,self.savedParameters[k]==v),0)
			

			# params = self.savedParameters.loc[name,:]
			params = self.savedParameters[select]
			assert params.shape[0] == 1, "Error, %d rows selected instead of 1!" % params.shape[0]
			# convert to series
			params = params.squeeze()

			kernel = _assignParametersToKernel(kernel,params)
			gvar = params.Gaussian_noise_variance

		inputColumns = self.inputColumns()

		gp = self.model(x,y,kernel,**self.model_kwargs)

		if optimize:
			gp.optimize()

			# save parameters is name provided and not already saved
			# if (not name is None) and not (name in self.savedParameters.index):
			# 	self.saveOptimizedParameters(gp,name)

		elif useSaved:
			gp = _assignGaussianNoiseVariance(gp,gvar)

		return gp

	def saveOptimizedParameters(self,gp,name=None,**kwargs):
		params = _get_params(gp,name,**kwargs)

		self.savedParameters = self.savedParameters.append(params)

	def addSavedParameters(self,params):
		self.savedParameters = self.savedParameters.append(params)

	def plot(self,edata,gp=None,name=None,output=None,logged=None,colspan=None,rowspan=None):
		# setup the gp
		if not name is None:
			gp = self.build(edata,name=name)
		cov = gp.kern.K(self.buildInput(edata))

		if colspan is None:
			colspan = 2

		# acutal plotting
		nrows = colspan + 1 + len(self.inputColumns())

		import matplotlib.pyplot as plt

		# fig = plt.figure(figsize=(9,3*nrows))
		fig = plt.figure()

		# plot input columns
		for ind,col in enumerate(self.inputColumns()):
			plt.subplot2grid((nrows,colspan+1),[ind,0],colspan=colspan)

			plt.plot(range(self.buildInput(edata).shape[0]),self.buildInput(edata)[:,ind])
			plt.ylim(min(self.buildInput(edata)[:,ind])-.5,max(self.buildInput(edata)[:,ind])+.5)
			plt.xticks([])
			plt.yticks([])
			plt.ylabel(col,fontsize=15)

		# plot optical density
		plt.subplot2grid((nrows,colspan+1),[ind+1,0],colspan=colspan)
		plt.plot(range(self.buildInput(edata).shape[0]),self.buildOutput(edata))
		plt.xticks([])
		plt.yticks([])
		plt.ylabel("OD",fontsize=15)

		# plot kernel function
		plt.subplot2grid((nrows,colspan+1),[ind+2,0],rowspan=colspan,colspan=colspan)
		if logged:
			cov = np.log(cov)
		im = plt.imshow(cov,interpolation="none",cmap="Blues",vmin=0)
		plt.xticks([])
		plt.yticks([])
		plt.grid(False)
		plt.ylabel("Kernel function",fontsize=15)

		ax = plt.subplot2grid((nrows,colspan+1),[ind+2,colspan],rowspan=colspan)
		plt.colorbar(im)
		ax.set_visible(False)

		if not output is None:
			plt.savefig(output)

	def __repr__(self,):
		ret = "Gaussian process factory\n"
		for n,k,p in self.inputDimensions:
			ret += "\t%s, (%s)\n" % (n,str(k))

		if self.savedParameters.shape[0] > 0:
			ret += "\n"
			ret += self.savedParameters.__repr__()

		return ret

class ProductFactory(Factory):

	"""Factory class that allows a hieararchy of addition and multiplication operations on kernel dimensions.

	Attributes:
		multiplicationDimensions: list that defines the addition and multiplication operations to apply to kernel dimensions.
	"""

	def __init__(self,normalize=False):
		Factory.__init__(self,normalize)

		# hieararchy of mult,add,mult,add... layers to generate kernels
		self.multiplicationDimensions = []

	def buildKernel(self,):

		# columns to use for x input to GP
		input_columns = self.inputColumns()

		def buildKernel_recursion(item,switch=1,**kwargs):

			switch = switch % 2

			if type(item) == list:
				k = buildKernel_recursion(item[0],switch+1)
				for i in item[1:]:
					if switch == 1:
						k += buildKernel_recursion(i,switch+1)
					else:
						k *= buildKernel_recursion(i,switch+1)
				return k
			else:
				if item is None:
					kernel = GPy.kern.Bias(1)
				else:
					ind = input_columns.index(item)
					ktype = self.inputDimensions[ind][1]
					kw = self.inputDimensions[ind][2]
					kernel = ktype(1,active_dims=[ind],name = item,**kw)
				return kernel

		k = buildKernel_recursion(self.multiplicationDimensions)

		return k

	def setMultiplicationDimension(self,multdims):

		self.multiplicationDimensions = multdims


def _get_params(gp,name=None,**kwargs):

	if name is None:
		name = ""

	params = []
	param_names = []
	ret = {}

	# param_names.append("log-likelihood")
	# params.append(gp.log_likelihood())
	ret['log-likelihood'] = gp.log_likelihood()

	# kernel parameters
	for p in gp.kern.parameters:
		if len(p.parameter_names()) > 1:
			for n in p.parameter_names():
				# param_names.append("_".join([p.name,n]))
				# params.append(p[n][0])
				ret["_".join([p.name,n])] = p[n][0]
		else:
			ret[p.name] = p[0]
    
    # likelihood parameters (gaussian noise)
	for p in gp.likelihood.parameters:
		# need to go deeper
		if isinstance(p,GPy.likelihoods.likelihood.Likelihood):
			for t in p.parameters:
				for ind,n in enumerate(t.parameter_names()):
					# param_names.append("_".join([gp.likelihood.name,n]))
					# params.append(p.values[ind])
					ret["_".join([gp.likelihood.name,p.name,n])] = t.values[ind]
		else:
			for ind,n in enumerate(p.parameter_names()):
				# param_names.append("_".join([gp.likelihood.name,n]))
				# params.append(p.values[ind])
				ret["_".join([gp.likelihood.name,n])] = p.values[ind]

	for k,v in kwargs.iteritems():
		ret[k] = v

	#return pd.Series(params,index=param_names)
	return pd.DataFrame(ret,index=[name])


def _assignParametersToKernel(k,params):
	for p in k.parameters:
		for n in p.parameter_names():

			if p.name != n:
				search = "_".join([p.name, n])
				temp = params[params.index==search]

				# if temp.shape[0] > 1:
				# 	temp = temp.mean()

				p[n] = temp
			else:
				p[0] = params[params.index==p.name]

	return k

def _assignGaussianNoiseVariance(gp,gvar):
	gp.likelihood.variance = gvar
	return gp
