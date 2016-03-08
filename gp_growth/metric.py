import numpy as np
from scipy import stats
from data import growth

#########
# utility functions
########

def gpDerivative(x,gp):

	if x.ndim == 1:
		x = x[:,np.newaxis]

	# from Solak et al.
	mu,ignore = gp.predictive_gradients(x)
	ignore,cov = gp.predict(x,full_cov=True)
	mult = [[((1./gp.kern.lengthscale)*(1-(1./gp.kern.lengthscale)*(y - z)**2))[0] for y in x] for z in x]
	return mu, mult*cov

def simMax(mu,cov,n=10):
	while mu.ndim > 1:
		mu = mu[:,0]

	sample = np.random.multivariate_normal(mu,cov,n)
	max_est = np.max(sample,1)

	return max_est


######
# Growth Metrics
######

class GrowthMetric(object):
	"""Abstract class defining growth metric calculations (e.g. AUC, mu max) on a GP model of cell growth.
	"""

	# def __init__(self, f, m, xNeeded = True, outputDim=None, **kwargs):
	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		self.predictive_data = predictive_data
		self.model = model
		self.factory = factory
		self.training_data = training_data
		self.factory_params = factory_params
		self.predictive_data_params = predictive_data_params

	def inputValid(self,predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		if not self.predictive_data is None:
			predictive_data = self.predictive_data
		if not self.model is None:
			model = self.model
		if not self.factory is None:
			factory = self.factory
		if not self.training_data is None:
			training_data = self.training_data
		if not self.factory_params is None:
			factory_params = self.factory_params
		if not self.predictive_data_params is None:
			predictive_data_params = self.predictive_data_params

		if not predictive_data is None:
			if model:
				return True
			elif factory:
				if not training_data is None:
					return True
		elif predictive_data_params and factory:
			if model:
				return True
			if not training_data is None:
				return True

		return False

	def buildModel(self,predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		if not self.predictive_data is None:
			predictive_data = self.predictive_data
		if not self.model is None:
			model = self.model
		if not self.factory is None:
			factory = self.factory
		if not self.training_data is None:
			training_data = self.training_data
		if not self.factory_params is None:
			factory_params = self.factory_params
		if not self.predictive_data_params is None:
			predictive_data_params = self.predictive_data_params

		if model:
			return model

		thinning = None
		if "thinning" in kwargs:
			thinning = kwargs['thinning']

		if factory:
			if isinstance(training_data,growth.GrowthData):
				training_data = training_data.getData("gp",thinning=thinning)

			if factory_params:
				model = factory.build(training_data,useSaved=True,optimize=False,**factory_params)
			else:
				model = factory.build(training_data,optimize=True)

		return model

	def buildPredictiveData(self,predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		if not self.predictive_data is None:
			predictive_data = self.predictive_data
		if not self.model is None:
			model = self.model
		if not self.factory is None:
			factory = self.factory
		if not self.training_data is None:
			training_data = self.training_data
		if not self.factory_params is None:
			factory_params = self.factory_params
		if not self.predictive_data_params is None:
			predictive_data_params = self.predictive_data_params

		thinning = None
		if "thinning" in kwargs:
			thinning = kwargs['thinning']

		if not predictive_data is None:
			if isinstance(predictive_data,growth.GrowthData):
				predictive_data = predictive_data.getData("gp",thinning=thinning)
				predictive_data = factory.buildInput(predictive_data)
			return predictive_data

		if factory and predictive_data_params:
			predictive_data = factory.buildInputFixed(**predictive_data_params)

		return predictive_data

	def compute(self,predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		assert self.inputValid(predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs), "Must provide valid input!"

		model = self.buildModel(predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)
		predictive_data = self.buildPredictiveData(predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

		assert model, "model is None!"
		assert not predictive_data is None, "predictive_data is None!"

		ret =  self._compute(predictive_data,model,**kwargs)

		# cleanup
		del model
		del predictive_data
		
		return ret

	def _compute(self,predictive_data,model,**kwargs):
		raise NotImplemented("must implement your growth metric's compute function!")

	def __call__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None, **kwargs):
		return self.compute(predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

class AUC(GrowthMetric):

	# DEFAULT_X_SHAPE = 100

	# def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
	# 	GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

	def _compute(self, predictive_data, model, **kwargs):
		
		mu,cov = model.predict(predictive_data,full_cov=True)
		dt = np.mean(predictive_data[1:,0]-predictive_data[:-1,0])
		D = np.repeat(dt,mu.shape[0]).T

		mu = np.dot(D,mu)[0]
		var = np.dot(D,np.dot(cov,D))
		return mu,var

class GrowthRate(GrowthMetric):

	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

	def _compute(self,predictive_data, model, **kwargs):

		mu,cov = gpDerivative(predictive_data,model)
		return mu, cov

class MuMax(GrowthMetric):

	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,n=None,**kwargs):
		GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

		if n is None:
			self.n = 10
		else:
			self.n = n

	def _compute(self,predictive_data, model, **kwargs):

		mu,cov = gpDerivative(predictive_data,model)
		sim = simMax(mu[:,0,0],cov,n=self.n)

		if 'plot' in kwargs and kwargs['plot']:
			import matplotlib.pyplot as plt
			plt.plot(mu[:,0,0])

		return np.mean(sim), np.std(sim)

class MuMax_simple(GrowthMetric):

	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

	def _compute(self,predictive_data, model, **kwargs):

		mu,cov = gpDerivative(predictive_data,model)
		ind = np.where(mu==mu.max())[0]

		return mu[ind,0],np.diag(cov)[ind][0]

class CarryingCapacity(GrowthMetric):

	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,n=None,**kwargs):
		GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)

		if n is None:
			self.n = 10
		else:
			self.n = n

	def _compute(self,predictive_data, model, **kwargs):

		mu,cov = model.predict(predictive_data,full_cov=True)
		sim = simMax(mu,cov,n=self.n)
		return np.mean(sim), np.std(sim)


class CarryingCapacity_simple(GrowthMetric):

	def __init__(self, predictive_data=None, model=None, factory=None, training_data=None, factory_params=None, predictive_data_params=None,**kwargs):
		GrowthMetric.__init__(self, predictive_data, model, factory, training_data, factory_params, predictive_data_params,**kwargs)


	def _compute(self,predictive_data, model, **kwargs):

		mu,cov = model.predict(predictive_data,full_cov=True)
		ind = np.where(mu==mu.max())[0]

		return mu[ind,0],np.diag(cov)[ind][0]


class LagTime(GrowthMetric):

	def __init__(self,f,threshold=None,index=None,*args,**kwargs):
		GrowthMetric.__init__(self,f,*args,**kwargs)

		self.index = index
		if self.index is None:
			self.index = 0

		if threshold is None:
			self.threshold = .95
		else:
			self.threshold = threshold


	def _compute(self, x, gp ):

		mu,var = gpDerivative(x,gp)

		prob = np.array([stats.norm.cdf(0,loc=m,scale=np.sqrt(v))[0] for m,v in zip(mu[:,:,0],var[:,0])])

		ind = 0
		while ind < prob.shape[0] and prob[ind] >self.threshold:
			ind += 1
		if ind == prob.shape[0]:
			ind -= 1
		return x[ind]

class MeanSquareError(GrowthMetric):

	# def __init__(self,f):
	# 	GrowthMetric.__init__(self,factory=f)

	def _compute(self,x,gp):

		n = gp.X.shape[0]
		mse = 1./n * sum(((gp.Y - gp.predict(gp.X)[0])**2)[:,0],)

		return mse