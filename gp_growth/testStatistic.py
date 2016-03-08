import numpy as np
import scipy.stats
import normal, GPy

class TestStatistic(object):

	@staticmethod
	def buildTestMatrix(n):
		A = np.zeros((n-1,2*n))
		A[:,0] = 1
		A[range(n-1),range(1,n)] = -1
		A[:,n] = -1
		A[range(n-1),n+np.arange(1,n)] = 1

		return A

	def __init__(self,data,additionalDimensions=None,normalize=False):
		if additionalDimensions is None:
			additionalDimensions = []

		self.additionalDimensions = additionalDimensions
		self.inputDimensions = ['time','Strain'] + additionalDimensions
		self.data = data
		self.k = len(self.inputDimensions)
		self.normalize = normalize

		self.gp = None
		self.mean, self.std = None,None
		self.means = {}
		self.stds = {}


	def buildData(self,thin=None,*args,**kwargs):
		if thin is None:
			thin = 8

		data = self.data
		# normalize by the global mean/variance
		self.mean, self.std = data.data.iloc[:,1:].values.mean(),data.data.iloc[:,1:].values.std()
		data.data.iloc[:,1:] = (data.data.iloc[:,1:] - self.mean)/self.std

		edata = data.getData("gp",thinning=thin)
		edata.Strain = edata.Strain == "ura3"

		edata = self.dataPostProcess(edata)

		if self.normalize:
			for name in self.inputDimensions:
				self.means[name] = edata[name].mean()
				self.stds[name] = edata[name].std()
				edata[name] = (edata[name] - self.means[name])/self.stds[name]

		return edata

	def dataPostProcess(self,edata):
		"""overide this to apply some change to GP input data"""
		return edata

	def buildKernel(self,):
		return GPy.kern.RBF(self.k,ARD=True)

	def buildGP(self,*args,**kwargs):
		if self.gp is None:
			edata = self.buildData(*args,**kwargs)

			gp = GPy.models.GPRegression(edata[self.inputDimensions].values,edata.od.values[:,None],self.buildKernel())
			gp.optimize()

			self.gp = gp

		return self.gp

	def predict(self,t,function=True,*args,**kwargs):
		"""
		return the distribution of (y_m(t_k) - y_m(t_0)) - (y_p(t_k) - y_p(t_0))
		"""
		n = t.shape[0]

		xpred = np.zeros((n,self.k))
		xpred[:,0] = t
		if self.normalize:
			xpred[:,0] = (xpred[:,0]-self.means['time'])/self.stds['time']

		xpred[:,1] = 1 # ura3 indicator

		for ad in self.inputDimensions:
			if ad in kwargs:
				ind = self.inputDimensions.index(ad)
				if self.normalize:
					xpred[:,ind] = (kwargs[ad]-self.means[ad])/self.stds[ad]
				else:
					xpred[:,ind] = kwargs[ad]

		gp = self.buildGP(*args,**kwargs)

		# TODO: need to double check _raw_predict gives me the correct output (latent function posterior)
		if function:
			mu,cov = gp._raw_predict(xpred,full_cov=True)
		else:
			mu,cov = gp.predict(xpred,full_cov=True)
		mvn = normal.MultivariateNormal(mu[:,0],cov)

		# rescale by mean, std
		mvn = mvn.dot(np.diag(np.ones(mvn.n))*self.std)
		mvn.mean+= self.mean

		return mvn

	def computeFullDifference(self,t,*args,**kwargs):
		"""
		return the distribution of (y_m(t_k) - y_m(t_0)) - (y_p(t_k) - y_p(t_0))
		"""
		n = t.shape[0]

		xpred = np.zeros((n*2,self.k))
		xpred[:n,0] = t
		xpred[n:,0] = t

		xpred[:n,1] = 1 # ura3 indicator

		if self.normalize:
			xpred[:,0] = (xpred[:,0]-self.means['time'])/self.stds['time']
			xpred[:,1] = (xpred[:,1]-self.means['Strain'])/self.stds['Strain']

		for ad in self.inputDimensions:
			if ad in kwargs:
				ind = self.inputDimensions.index(ad)
				if self.normalize:
					xpred[:,ind] = (kwargs[ad]-self.means[ad])/self.stds[ad]
				else:
					xpred[:,ind] = kwargs[ad]

		gp = self.buildGP(*args,**kwargs)

		# TODO: need to double check _raw_predict gives me the correct output (latent function posterior)
		mu,cov = gp._raw_predict(xpred,full_cov=True)
		funcMVN = normal.MultivariateNormal(mu[:,0],cov)

		A = self.buildTestMatrix(n)
		diffMVN = funcMVN.dot(A)

		return diffMVN

	def computeFullConfidence(self,t,*args,**kwargs):
		"""
		return the probability that (y_m(t_k) - y_m(t_0)) - (y_p(t_k) - y_p(t_0)) > 0
		"""
		diffMVN = self.computeFullDifference(t,*args,**kwargs)

		return [scipy.stats.norm.cdf(0,loc=m,scale=s) for m,s in zip(diffMVN.mean.tolist(),np.sqrt(np.diag(diffMVN.cov)).tolist())]

	def computeDifferenceIntegral(self,t,*args,**kwargs):
		diffMVN = self.computeFullDifference(t,*args,**kwargs)

		B = np.ones(diffMVN.n)
		intMVN = diffMVN.dot(B)

		return intMVN

	def computeIntegralConfidence(self,t,*args,**kwargs):
		intMVN = self.computeDifferenceIntegral(t,*args,**kwargs)
		return scipy.stats.norm.cdf(0,loc=intMVN.mean[0],scale=np.sqrt(intMVN.cov[0,0]))
