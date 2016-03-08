import numpy as np
from scipy.optimize import curve_fit

def gompertz(t,m,A,l):
    return A * np.exp(-np.exp(m*np.e/A*(l-t) + 1))

def meanSquareError(time,od):
	# Gompertz Testing
	try:
		popt, pcov = curve_fit(gompertz,time,od)

		m,A,l = popt

		if m <= 0:
			raise RuntimeError()

		predict = np.array([gompertz(t,*popt) for t in time])
		mse = 1./n * sum(((od - predict)**2),)
	except RuntimeError, e:
		mse = 1./n * sum(((od - np.mean(od))**2),)

	return mse

def optimize(time,od):

	time = time.astype(float)
	od = od.astype(float)
	n = time.shape[0]

	try:
		popt, pcov = curve_fit(gompertz,time,od)

		m,A,l = popt

		if m <= 0:
			raise RuntimeError()

		predict = np.array([gompertz(t,*popt) for t in time])
		mse = 1./n * sum(((od - predict)**2),)
	except RuntimeError, e:
		m,A,l = np.nan, np.nan, np.nan
		mse = 1./n * sum(((od - np.mean(od))**2),)

	return [m,A,l,mse]