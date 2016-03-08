import numpy as np
import pandas as pd
import os
import itertools
import time
import datetime
import re
import matplotlib.pyplot as plt

########################################################
## Plotting functions
########################################################

def gauss_plot(x,mean,var=None,line_col="g",fill_col="g",label=""):

	if not var is None and not mean.shape == var.shape:
		print "Error mean and variance not same shape (",str(mean.shape),",",str(var.shape),")"
		return

	plt.plot(x,mean,c=line_col,label=label)
	
	if not var is None:
		plt.fill(np.concatenate([x, x[::-1]]),
		np.concatenate([mean - 1.9600 * np.sqrt(var),
		               (mean + 1.9600 * np.sqrt(var))[::-1]]),
		alpha=.5, fc=fill_col, ec='None', label='')

########################################################
## Data functions
########################################################

def load_iron(data,mode=1,subsample=-1):
	data = pd.read_csv(data)
	
	# extract metadata from columns
	cols = data.columns.to_series()
	cols = cols.str.replace("\xce\xbc","micro")
	cols = cols.str.encode("ascii","replace")
	
	offset = 1
	conc_ind = 0
	col_names = ["microMolar Fe","strain"]
	if mode == 1:
		cols = cols.str.extract("\(([0-9]*)[a-zA-Z-?_ ]*\)([a-zA-Z1-9]*)")
	else:
		cols = cols.str.extract("([a-zA-Z1-9]*)\(([0-9]*)[a-zA-Z-? ]*\)")
		cols.replace("B","blank",inplace=True)
		offset = 2
		conc_ind=1
		col_names = ["strain","microMolar Fe"]
	
	cols = cols.iloc[offset:,:]	
	cols = cols.replace('','0')
	cols.iloc[:,conc_ind] = cols.iloc[:,conc_ind].astype(float)
	cols = cols.rename(columns=pd.Series(col_names))
	
	od = data.iloc[:,offset:].T
	od = np.log2(od.divide(od.iloc[:,0],"rows"))
	time = data.Time

	if not subsample == -1:
		select = np.arange(0,time.shape[0],subsample)
		# print select
		# ind = np.apply_along_axis(lambda t: min(abs(t-select))<1e-9,1,time[:,np.newaxis])
		# od = od.iloc[:,ind]
		# time = time[ind]
		od = od.iloc[:,select]
		time = time[select]

	group = cols.groupby(["strain","microMolar Fe"])
	#replicatify = lambda x: range(x.shape[0])
	#print group.transform(replicatify)
	
	cols['replicate'] = 0
	for exp,temp in group:
		#temp['replicate'] = range(temp.shape[0])
		cols.loc[group.groups[exp],'replicate'] = range(temp.shape[0])
		#print exp
		#print temp

	edata = expand_data(time,od,cols)
	edata = metadata_cartesian_product(edata,['strain'],removeOld=True)

	return edata

# given an array X and columns,
# create cartesian product of all combinations of values in each column
# If hierarchy is true, follows the hierarchy implied by the column order
# and higher level columns get a seperate output column
def metadata_cartesian_product(X,columns,hierarchy=False,prefix=False,removeOld=False):

	X = X.copy()
	temp = X[columns]
	n_columns = temp.shape[1]
	
	conditions = [np.unique(temp.values[:,i]) for i in range(n_columns)]
	conditions = list(itertools.product(*conditions))

	for cond in conditions:
		
		if prefix:
			names = ["".join([str(z) for z in y]) for y in zip(columns,cond)]
		else:
			names = [str(x) for x in cond]

		X["_".join(names)] = np.all(temp.eq(cond),1).astype(np.int8)

		if hierarchy:
			for i in range(n_columns-1):
				X["_".join(names[:i+1])] = np.all(temp.values[:,:i+1] == cond[:i+1],1).astype(np.int8)

	if removeOld:
		for c in columns:
			del X[c]
				
	return X

def metadata_condition_parse(cond,defaultValue=0):
	
	if any(cond.str.contains("H2O2")):
		if any(cond.str.contains("+")):
			delim = "+"
		else:
			delim = " "
		suffix = "mM H2O2"
	elif any(cond.str.contains("shift")):
		delim = " "
		suffix = "C shift"
	else:
		delim = " "
		suffix = "unknown"

	cond = cond.str.rstrip(suffix)
	split = cond.str.split(delim)
	print split
	ret = pd.DataFrame(split.apply(lambda x: delim.join(x[:-1])))
	ret = ret.rename(columns={'Condition':'media'})
	ret[suffix] = split.apply( lambda x: float(x[-1].rstrip(suffix)) if len(x) > 1 and suffix in x[-1] else defaultValue )

	return ret


def expand_data(time,od,params=None):

	ret = pd.DataFrame()
	for i in range(od.shape[0]):
		if len(ret) == 0:
			if not params is None:
				ret = expand_row(time,od.iloc[i,:],params.iloc[i,:])
			else:
				ret = expand_row(time,od.iloc[i,:],None)
		else:
			if not params is None:
				ret = pd.concat((ret,expand_row(time,od.iloc[i,:],params.iloc[i,:])))
			else:
				ret = pd.concat((ret,expand_row(time,od.iloc[i,:],None)))
	return ret

# function for expanding row for GP regression
def expand_row(time,od,params=None):
	ret = pd.DataFrame()
	ret['time'] = time.astype(float)
	ret['od'] = od.values
	
	if not params is None:
		for c,v in params.iteritems():
		    ret[c] = v

	return ret

def expand_data_row(r):
	well = int(r.name)
	r = pd.DataFrame(r)
	r.columns = ["od"]
	r['Well'] = well
	r['time'] = r.index

	return r

def parse_time(t):
	try:
		return time.struct_time(time.strptime(t,'%H:%M:%S'))
	except ValueError, e:
		try:
			t = time.strptime(t,'%d %H:%M:%S')
			t = list(t)
			t[2]+=1
			return time.struct_time(t)
		except ValueError, e:
			raise Exception("Time format unknown")

def convert_encoding(f,encoding,outcoding):
	open(f+".temp","w").write(open(f).read())
	os.popen('iconv -f '+encoding+' -t '+outcoding+' "'+ f +'.temp" > "' + f + '"')
	os.remove(f+'.temp')
	
def load_bioscreen(folder,convert=False,removeBlank=True,removeEmpty=True):

	files = os.listdir(folder)

	key_file = filter(lambda x: "key.xlsx" in x, files)
	data_file = filter(lambda x: ".csv" in x, files)

	assert len(data_file)==1, "No data file or more than one data file: "+ str(data_file)
	assert len(key_file)==1, "No key file or more than one key file: "+ str(key_file)

	data_path = os.path.join(folder,data_file[0])
	key_path = os.path.join(folder,key_file[0])

	# convert to UTF-8
	if convert:
		#open(data_path+".temp","w").write(open(data_path).read())
		#os.popen('iconv -f UTF-16 -t UTF-8 "'+ data_path+'.temp" > "' + data_path + '"')
		#os.remove(data_path+'.temp')
		convert_encoding(data_path,"UTF-16","UTF-8")

	key = pd.read_excel(key_path)
	data = pd.read_csv(data_path)

	key = key.replace(np.nan,"")

	# align key and data
	select = data.columns[1:].isin(key.Well.astype(str))
	data = data.iloc[:,[0] + (np.where(select)[0]+1).tolist()]

	select = key.Well.astype(str).isin(data.columns[1:])
	key = key.iloc[np.where(select)[0].tolist(),:]


	def convert_time(x):
		delta = datetime.datetime(*x[:-2]) - datetime.datetime(*t[0][:-2])
		return 24*delta.days + float(delta.seconds)/3600

	# convert time
	t = data['Time'].apply(parse_time)
	#t = t.apply(lambda x: (time.mktime(x) - time.mktime(t[0]))/3600).round(2)
	#t = t.apply(lambda x: float((datetime.datetime(*x[:-2]) - datetime.datetime(*t[0][:-2])).seconds)/3600).round(2)
	t = t.apply(convert_time).round(2)
	data['Time'] = t

	# subtract and remove bioscreen blank
	#data.iloc[:,2:] = data.iloc[:,2:].subtract(data['Blank'],0)
	#del data['Blank']
	if removeBlank:
		# od = data.iloc[:,1:]
		# od = od.iloc[:,key.Strain.values!="blank"]
		# data.iloc[:,1:] = od
		# key = key[key.Strain!="blank"]
		data = data.drop(data.columns[1:][key.Strain.values == "blank"],axis=1)
		key = key.drop(key.index[key.Strain== "blank"])

	if removeEmpty:
		data = data.drop(data.columns[1:][key.Strain.values == ""],axis=1)
		key = key.drop(key.index[key.Strain== ""])

		

	# divide by firt time point
	# data.iloc[:,1:] = data.iloc[:,1:].divide(data.iloc[0,1:],1)

	# log od's
	# data.iloc[:,1:] = data.iloc[:,1:].apply(np.log)

	# assign batch label
	batch = folder
	batch = batch.rstrip("/")
	batch = batch[batch.rfind("/")+1:]
	key['batch'] = batch

	return key,data

def plot_bioscreen(key,data,title_index=[0],fixed_y=True,output=""):
	condition = key.columns
	condition = condition[condition!="Well"]
	condition = condition[condition!="Bio"]
	condition = condition[condition!="Tech"]

	groups = key.groupby(condition.tolist())

	time = data.iloc[:,0]
	od = data.iloc[:,1:]

	plt.figure(figsize=(5*4,groups.ngroups/5*4))

	for i,val in enumerate(groups.groups.iteritems()):
		k,ind = val
		print k,ind
		ax = plt.subplot(groups.ngroups/5+1,5,i+1)
		temp_key = groups.get_group(k)
		temp_data = od.ix[:,temp_key.Well.astype(str)]
		# for b in temp_key.Bio.unique():
		# 	temp2 = temp_data.iloc[:,np.where(temp_key.Bio==b)[0]]
		# 	temp2.plot(x=time,ax=ax,legend=False)
		if fixed_y:
			temp_data.plot(x=time,ax=ax,legend=False,ylim=(min(od.min()),max(od.max())))
		else:
			temp_data.plot(x=time,ax=ax,legend=False)

		ax.set_title(" ".join(str(k[j]) for j in title_index))

	plt.tight_layout()

	if output == "":
		plt.show()
	else:
		plt.savefig(output)
		plt.close()


def expand_bioscreen(key,data):

	time_ind  = key.shape[1]

	temp = data.iloc[:,1:]
	temp.index = data.Time
	temp.columns = temp.columns.astype(np.int64)

	combine = pd.merge(key,temp.T,left_on="Well",right_index=True)

	# subtract blank values
	blank = combine[combine.Strain=="blank"].iloc[:,time_ind:].mean()
	combine.iloc[:,time_ind:] = combine.iloc[:,time_ind:] - blank


	# expand rows
	combine = combine[combine.Strain!="blank"]
	r = combine.iloc[0,:]
	expand_data = expand_data_row(r.iloc[time_ind:])
	# expand_data['Well'] = r.Well
	# expand_data['Strain'] = r.Strain
	# expand_data['Bio'] = r.Bio
	# expand_data['Tech'] = r.Tech
	# expand_data['Media'] = r.Media
	# expand_data['Condition'] = r.Condition
	# expand_data['batch'] = r.batch
	for c in r.index[:time_ind]:
			expand_data[c] = r[c]

	for i in range(1,combine.shape[0]):
		r = combine.iloc[i,:]
		temp = expand_data_row(r.iloc[time_ind:])
		# temp['Well'] = r.Well
		# temp['Strain'] = r.Strain
		# temp['Bio'] = r.Bio
		# temp['Tech'] = r.Tech
		# temp['Media'] = r.Media
		# temp['Condition'] = r.Condition
		# temp['batch'] = r.batch
		for c in r.index[:time_ind]:
			temp[c] = r[c]
		#temp[r.index[:time_ind]] = r[:time_ind]
		expand_data = expand_data.append(temp)


	# expand rows with key metadata
	# data = data.set_index('Time')
	# data = data.T
	# temp = expand_data_row(data.iloc[0,:])
	# for i in range(1,data.shape[0]):
	# 	temp = temp.append(expand_data_row(data.iloc[i,:]))
	# expand_data = temp
	# expand_data = pd.merge(key,expand_data)

	# remove blank rows
	expand_data = expand_data[expand_data['Strain'] != 'blank']

	# concatenate cartesian product of condition to data frame
	#cond = expand_data['Condition']
	#expand_data = pd.concat((expand_data,
	#	metadata_cartesian_product(metadata_condition_parse(cond),columns=["media"],removeOld=True)),
	#	1)

	# concatenate cartesian product of strain to data frame
	#expand_data = metadata_cartesian_product(expand_data,columns=["Strain"],removeOld=False)

	# concatenate cartesian product of bio/tech rep to data frame
	#expand_data = metadata_cartesian_product(expand_data,columns=["batch","Strain","Bio","Tech"],prefix=False,hierarchy=False,removeOld=True)

	# Booleanize variables
	expand_data = metadata_cartesian_product(expand_data,columns=["Strain"],removeOld=True)
	expand_data = metadata_cartesian_product(expand_data,columns=["batch"],removeOld=True)
	expand_data = metadata_cartesian_product(expand_data,columns=["Bio"],prefix=True,removeOld=True)
	expand_data = metadata_cartesian_product(expand_data,columns=["Tech"],prefix=True,removeOld=True)

	return expand_data

def load_all_bioscreen(folders,convert=False):
	print folders[0]
	key,data = load_bioscreen(folders[0],convert)
	expand_data = expand_bioscreen(key,data)
	for folder in folders[1:]:
		print folder
		key,data = load_bioscreen(folder,convert)
		expand_data = expand_data.append(expand_bioscreen(key,data))

	expand_data = expand_data.replace(np.nan,0)

	return expand_data

def convert_bioscreen_key(f,condition,**kwargs):
	key = pd.read_excel(f)

	if condition == "H2O2":
		mm = key.Condition.str.rstrip("mM H2O2").str.strip("CM").str.strip("+mev")
		mm = mm.replace("",0).astype(int).values
		key['mM H2O2'] = mm
		key['Media'] = np.where(key.Condition.str.contains("mev"),"CM+mev","CM")
	elif condition == "Heat":
		key['Heat Shift C'] = 56
		key['Heat Shift Time'] = 16
		key['Media'] = np.where(key.Condition.str.contains("mev"),"CM+mev","CM")

	for k,v in kwargs.iteritems():
		key[k] = v

	key.to_csv(f.rstrip(".xls")+".csv",index=False)
