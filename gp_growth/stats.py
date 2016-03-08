import numpy as np

def mvn_chisquare(x,mu,cov):
	"""return the chisquare statistic of a multivariate normal sample

	see http://www.nr.com/CS395T/lectures2010/2010_6_MultivariateNormalAndChiSquare.pdf"""

	L = np.linalg.cholesky(cov)
	y = np.linalg.solve(L,x-mu)
	return np.dot(y,y)