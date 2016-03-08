from GPy.kern import Kern,RBF
from GPy.core.parameterization.transformations import Logexp
from GPy.core.parameterization import Param
import numpy as np

class Categorical(Kern):

	def __init__(self, input_dim,variance=1,active_dims=[0],name="categorical", inverse=False,useGPU=False):
		super(Categorical, self).__init__(input_dim, active_dims, name,useGPU=useGPU)

		self.inverse = inverse

		self.variance = Param('variance',variance,Logexp())
		self.link_parameter(self.variance)

	def K(self,X,X2=None):
		if X2 is None:
			X2 = X

		if self.inverse:
			return self.variance * np.where(X == X2.T,0,1)
		else:
			return self.variance * np.where(X == X2.T,1,0)

	def Kdiag(self,X):
		if self.inverse:
			return np.zeros(X.shape[0])
		return self.variance*np.ones(X.shape[0])

	def update_gradients_full(self, dL_dK, X, X2=None):
		if X2 is None:
			X2 = X

		if self.inverse:
			self.variance.gradient = np.nansum(dL_dK * np.where(X == X2.T,0,1))
		else:
			self.variance.gradient = np.nansum(dL_dK * np.where(X == X2.T,1,0))

	def update_gradients_diag(self, dL_dKdiag, X):
		print "update_gradients_diag"
		self.variance.gradient = np.sum(dL_dKdiag * np.ones(X.shape[0]))

	def gradients_X(self, dL_dK, X, X2):
		print "gradients_X"
		if X2 is None:
			X2 = X
		return dL_dK * np.where(X == X2.T,1,0)
		
	def gradients_X_diag(self, dL_dKdiag, X):
		print "gradients_X_diag"
		return dL_dKdiag * np.ones(X.shape[0])