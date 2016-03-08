import numpy as np
import warnings
import scipy

class MultivariateNormal(object):

	def __init__(self,mean,cov):
		self.mean = mean
		self.n = self.mean.shape[0]
		self.cov = cov
		if self.cov.ndim == 1:
			self.cov = np.diag(self.cov)

		# some checks
		if self.mean.ndim > 1:
			warnings.warn("mean multidimensional, taking first dimension")
			self.mean = self.mean[:,0]
		assert self.cov.ndim == 2, "covariance must be two dimensional"
		assert self.cov.shape[0] == self.n and self.cov.shape[1] == self.n, 'covariance shape does not match mean'

	def dot(self,A):
		if A.ndim == 1:
			A = A[None,:]

		assert A.shape[1] == self.n, 'input dimension must match n'
		m = np.dot(A,self.mean)
		c = np.dot(A,np.dot(self.cov,A.T))

		return MultivariateNormal(m,c)

	def quadratic(self,x):
		diff = x-self.mean
		return np.dot(diff,np.dot(np.linalg.inv(self.cov),diff.T))

	def add(self,other):
		if isinstance(other,MultivariateNormal):
			return MultivariateNormal(self.mean+other.mean,self.cov + other.cov)
		return None

	def subtract(self,other):
		if isinstance(other,MultivariateNormal):
			return MultivariateNormal(self.mean - other.mean,self.cov + other.cov)
		return None

	def plot(self,x=None,confidence=None,label="",color="b",alpha=.1):
		import matplotlib.pyplot as plt

		if x is None:
			x = range(self.n)
		if confidence is None:
			confidence = .95

		sigma = scipy.stats.norm.ppf(confidence)

		plt.plot(x,self.mean,c=color,label=label)
		plt.fill_between(x,self.mean+sigma*np.sqrt(np.diag(self.cov)),self.mean-sigma*np.sqrt(np.diag(self.cov)),alpha=alpha,color=color)
