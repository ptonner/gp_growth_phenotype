import numpy as np
from scipy.optimize import curve_fit

def gompertz(t,m,A,l):
    return A * np.exp(-np.exp(m*np.e/A*(l-t) + 1))

def logistic(t,m,A,l):
    return 1.*A / (1 + np.exp(4.*m/A*(l-t) + 2))

def richards(t,m,A,l,v):
    p = 1 + v*np.exp(1+v)*np.exp(1.*m/A*(1+v)*(1+1./v)*(l-t))
    return 1.*A * np.power(p,-1./v)

def meanSquareError(time,od,model=gompertz):
	# Gompertz Testing
	try:
		popt, pcov = curve_fit(model,time,od)

		m,A,l = popt[:3]

		if m <= 0:
			raise RuntimeError()

		predict = np.array([gompertz(t,*popt) for t in time])
		mse = 1./n * sum(((od - predict)**2),)
	except RuntimeError, e:
		mse = 1./n * sum(((od - np.mean(od))**2),)

	return mse

def optimize(time,od,model=gompertz):

	time = time.astype(float)
	od = od.astype(float)
	n = time.shape[0]

	try:
		popt, pcov = curve_fit(model,time,od)

		m,A,l = popt[:3]

		if m <= 0:
			raise RuntimeError()

		predict = np.array([model(t,*popt) for t in time])
		mse = 1./n * sum(((od - predict)**2),)
	except RuntimeError, e:
		m,A,l = np.nan, np.nan, np.nan
		mse = 1./n * sum(((od - np.mean(od))**2),)

	return [m,A,l,mse]
