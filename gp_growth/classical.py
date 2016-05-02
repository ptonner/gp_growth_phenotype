import numpy as np
from scipy.optimize import curve_fit

def gompertz(t,m,A,l):
    return A * np.exp(-np.exp(m*np.e/A*(l-t) + 1))

def logistic(t,m,A,l):
    return 1.*A / (1 + np.exp(4.*m/A*(l-t) + 2))

def richards(t,m,A,l,v):
    p = 1 + v*np.exp(1+v)*np.exp(1.*m/A*(1+v)*(1+1./v)*(l-t))
    return 1.*A * np.power(p,-1./v)

def schnute(t,m,A,l,a=.1,b=.1):
    return (1.*m*(1-b)/a) * np.power((1-(b*np.exp(a*l+1-b-a*t)))/(1-b),1./b)

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
		mse = np.nan# 1./n * sum(((od - np.mean(od))**2),)

	return mse

def optimize(time,od,model=gompertz,p0=None,cv=None):

	time = time.astype(float)
	od = od.astype(float)
	n = time.shape[0]

	if cv:
		train_time = time[cv[0]]
		train_od = od[cv[0]]

		test_time = time[cv[1]]
		test_od = od[cv[1]]

	else:
		train_time = time
		train_od = od

		test_time = time
		test_od = od


	try:
		popt, pcov = curve_fit(model,train_time,train_od,p0=p0)

		m,A,l = popt[:3]

		if m <= 0:
			raise RuntimeError()

		predict = np.array([model(t,*popt) for t in test_time])
		mse = 1./test_od.shape[0] * sum(((test_od - predict)**2),)
	except RuntimeError, e:
		m,A,l = np.nan, np.nan, np.nan
		mse = np.nan

	return [m,A,l,mse]
