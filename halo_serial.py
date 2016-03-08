import pandas as pd
from gp_growth import factory,metric,gompertz
from gp_growth.data.growth import GrowthData
from gp_growth.storage import mongo
import os
import numpy as np
# from matplotlib.pyplot import *
# from scipy.optimize import curve_fit
# from lib.gompertz import gompertz
# import seaborn as sns

# storage.open("data/hsal.h5")
db = mongo.MongoDB()
plates = db.getPlates()
output = pd.DataFrame()
gpFactory = factory.Factory()
mse = metric.MeanSquareError(factory=gpFactory)
muMax = metric.MuMax(factory=gpFactory,n=100)
carryingCapacity = metric.CarryingCapacity(factory=gpFactory)
lagTime = metric.LagTime(.5,factory=gpFactory,)

outputDir = "notebooks/H. salinarum KO library/"
if "halo_ko_serial.csv" in os.listdir(outputDir):
	output = pd.read_csv(outputDir+"halo_ko_serial.csv")

for p in plates:
	#wells = storage.getExperimentalDesigns("well",plates=[p])
	data = db.getData(plate=p)
	data.transform(log=True)
	data.transform(subtract=0)

	# for w in wells:
	for i in range(1,data.key.shape[0]):
		if output.shape[0] > 0 and sum(np.all((output.well == data.key.well.iloc[i-1],output.batch==p),1)) != 0:
			continue

		# data = storage.getData(plates=[p],well=w)
		temp = data.data.iloc[:,[0,i]]
		params = data.key.iloc[i-1,:]
		params = pd.DataFrame([params.values],columns=params.index,index=[params.name])

		temp = GrowthData(temp,params)

		print p,i,temp.data.shape

		edata = temp.getData("gp")

		# GP MSE
		gp = gpFactory.build(edata,optimize=True)
		
		params['final_od'] = edata.od[-1]
		params['gp_mse'] = mse.compute(edata,gp)
		params['gp_muMax_mean'],params['gp_muMax_std'] = muMax.compute(predictive_data=temp,model=gp)
		params['gp_CarryingCapacity_mean'],params['gp_CarryingCapacity_std'] = carryingCapacity.compute(predictive_data=temp,model=gp)
		# params['gp_lagTime'] = lagTime.compute(gpFactory.buildInputFixed(time_min=0,time_max=max(edata.time),size=200,convert=False),gp)
		params['gp_loglikelihood'] = [gp.log_likelihood()]

		# Gompertz MSE
		m,A,l,gmse = gompertz.optimize(edata.time,edata.od)
		params['gompertz_muMax'] = m
		params['gompertz_CarryingCapacity'] = A
		params['gompertz_lagTime'] = l
		params['gompertz_mse'] = gmse

		params['ss_tot'] = np.sum((edata.od - edata.od.mean())**2)/edata.shape[0]

		print params

		if output.shape[0]==0:
			# output = pd.DataFrame(params)
			# output = output.T
			output = params
		else:
			# output = output.append(params,ignore_index=True)
			output = output.append(params)


		output.to_csv(outputDir+"halo_ko_serial.csv",index=False)




# import pandas as pd
# import GPy
# from lib import utils,analysis
# from numpy import *
# import numpy as np
# from scipy.optimize import curve_fit
# from lib.gompertz import gompertz
# from lib import model_validation
# import time as libtime
# import sys,getopt


# output = pd.DataFrame()
# skip = pd.DataFrame()

# # check for non-zero starting index
# argv = sys.argv[1:]
# start_ind = 0
# opt,arg = getopt.getopt(argv,'s:')
# for o,a in opt:
# 	if o == "-s":
# 		start_ind = int(a)
# 		output = pd.read_csv("output/halo_ko/halo_ko_serial.csv")

# data = pd.read_csv("output/halo_ko/halo_ko_data.csv")
# time_ind = range(145)

# for i in range(start_ind,data.shape[0]):

# 	# if i > 10:
# 	# 	break

# 	row = data.iloc[i,max(time_ind)+1:]
# 	od = data.iloc[i,time_ind].values.astype(float)
# 	time = data.columns[time_ind].values.astype(float)

# 	time = time[~np.isnan(od)]
# 	od = od[~np.isnan(od)]

# 	if row.Strain == "blank":
# 		continue

# 	if i % 1 == 0:
# 		print i,libtime.asctime(),row.values

# 	try:
# 		popt, pcov = curve_fit(gompertz,time,od)
# 	except RuntimeError, e:
# 		print "Gompertz failure"
# 		if skip.shape[0]==0:
# 			skip = pd.DataFrame(columns=row.index)
# 			skip = skip.append(row)
# 		else:
# 			skip = skip.append(row)
# 		continue
# 		#p0 = (0,0,0)
# 		#popt, pcov = curve_fit(gompertz,time,od,p0=p0)

# 	predict = [gompertz(t,*popt) for t in time]
# 	resid = predict - od
# 	sse = sum(resid**2)
# 	sst = sum((od-mean(od))**2)
# 	r2 = 1-abs(sse/sst)
# 	if r2 < 0:
# 		r2 = 0

# 	row['gompertz_mu_max'] = popt[0]
# 	row['gompertz_carrying_capacity'] = popt[1]
# 	row['gompertz_lag_time'] = popt[2]
# 	row['gompertz_sse'] = sse
# 	row['gompertz_sst'] = sst
# 	row['gompertz_r2'] = r2

# 	gp = GPy.models.GPRegression(atleast_2d(time).T,atleast_2d(od).T)
# 	gp.optimize()

# 	pred_time = np.linspace(min(time),max(time),1000)

# 	# mu_max
# 	# from Solak et al.
# 	mu,ignore = gp.predictive_gradients(pred_time[:,newaxis])
# 	ignore,cov = gp.predict(pred_time[:,newaxis],full_cov=True)
# 	mult = [[((1./gp.kern.lengthscale)*(1-(1./gp.kern.lengthscale)*(y - x)**2))[0] for y in pred_time] for x in pred_time]

# 	mu_max = analysis.sim_max(mu[:,0], mult*cov,500)
# 	row['gp_mu_max_mean'] = np.mean(mu_max)
# 	row['gp_mu_max_variance'] = np.var(mu_max)

# 	# carrying capacity
# 	try:
# 		mu,cov = gp.predict(pred_time[:,newaxis],full_cov=True)
# 		carr_cap = analysis.sim_max(mu, cov,1000)
# 		row['gp_carrying_capacity_mean'] = np.mean(carr_cap)
# 		row['gp_carrying_capacity_variance'] = np.var(carr_cap)
# 	except linalg.linalg.LinAlgError,e:
# 		row['gp_carrying_capacity_mean'] = np.nan
# 		row['gp_carrying_capacity_variance'] = np.nan

# 	# log likelihood
# 	row['gp_loglikelihood'] = gp.log_likelihood()
	
# 	# cross validation error
# 	errors = model_validation.gp_regression_cv(atleast_2d(time).T,atleast_2d(od).T,10)
# 	row['gp_cv_mean'] = mean(errors)
# 	row['gp_cv_var'] = np.var(errors)

# 	# lag time
# 	row['gp_lag_time'] = analysis.lagtime_gp(pred_time[:,newaxis],gp=gp)[0]

# 	if output.shape[0]==0:
# 		output = pd.DataFrame(columns=row.index)
# 		output = output.append(row)
# 	else:
# 		output = output.append(row)

# 	output.to_csv("output/halo_ko/halo_ko_serial.csv",index=False)


# output.to_csv("output/halo_ko/halo_ko_serial.csv",index=False)
# skip.to_csv("output/halo_ko/halo_ko_serial_skip.csv",index=False)
