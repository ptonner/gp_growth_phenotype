import matplotlib.pyplot as plt
import seaborn as sns
import os, time, datetime
import pandas as pd
import numpy as np
import itertools

class GrowthData(object):

	"""
	Data structure for handling microbial growth data.

	There is an assumed column 'well' inside *key* that matches the column names (other than 'time') in *data* that corresponds to the matching well's raw data.
	This column is not necessary, as the rows in *key* are assumed to map directly to *data*, e.g. row 0 of *key* is column 1 of *data*, etc.

	Attributes:
      data (pd.DataFrame): n by (p+1) DataFrame (for n timepoints and p wells) that stores the raw optical density data, the first column is assumed to be time.
      key (pd.DataFrame): p by k DataFrame (for k experimental designs) that stores experimental designs for each well
	"""


	def __init__(self,data=None,key=None,logged=None):
		self.key = key
		self.data = data

		assert type(key) == pd.DataFrame, "key must be a pandas dataframe"
		assert 'well' in key.columns, "must provide well information!"

		if logged is None:
			self.logged = False
		else:
			self.logged = logged

		self.replicateColumns = None

	def partition(self,num_partitions):
		if self.key is None:
			return

		# total number of wells to partition
		wells = self.key.shape[0]

		# how big will the partitions be?
		partition_size = wells/num_partitions
		if not wells % num_partitions == 0:
			partition_size+=1

		# create a list of partition ids, and randomize their order
		partition_ind = np.repeat(np.arange(num_partitions),partition_size)
		order = np.random.choice(np.arange(partition_ind.shape[0]),wells,replace=False)
		partition_ind = partition_ind[order]

		self.key['partition'] = partition_ind

	def plot(self,output="",fixed_y=True,title_index=None,colorby=None,fixed_colors=True,groupby=None,maxgroups=25,max_legend=10,figHeight=4,figWidth=4,ncols=None,groupOrder=None,newFig=None,useAx=None,colors=None,legend_kwargs=None,**kwargs):
		if groupby:
			if not type(groupby) == list:
				groupby = [groupby]
			condition = groupby
		else:
			condition = self.key.columns
			# for col in self.replicateColumns:
			# 	condition = condition.drop(col)
			condition = condition.tolist()

		if title_index is None:
			title_index = range(len(condition))

		if newFig is None:
			newFig = True

		if ncols is None:
			ncols = 5

		if legend_kwargs is None:
			legend_kwargs = {}

		groups = self.key.groupby(condition)
		time = self.data.iloc[:,0]
		od = self.data.iloc[:,1:]

		if colorby:
			categories = tuple(self.key[colorby].unique())
			if colors is None:
				color_lookup = dict(zip(categories,sns.color_palette("hls", len(categories))))
			else:
				color_lookup = dict(zip(categories,colors))

		ncols = min(ncols,groups.ngroups)
		nrows = 1
		if groups.ngroups > 1:
			nrows = (groups.ngroups-1)/ncols + 1
		# plt.figure(figsize=(5*4,(groups.ngroups/5+1)*4))
		# plt.figure(figsize=(ncols*figWidth,(groups.ngroups/ncols+1)*figHeight))

		if newFig:
			plt.figure(figsize=(ncols*figWidth,nrows*figHeight))

		if groups.ngroups > maxgroups:
			print "Error, number of groups (%d) is greater than max (%d)." % (groups.ngroups,maxgroups)
			return

		groupList = list(groups.groups.iteritems())
		if groupOrder is None:
			groupOrder = range(len(groupList))
		for i,j in enumerate(groupOrder):
			val = groupList[j]

			k,ind = val
			# ax = plt.subplot(groups.ngroups/5+1,5,i+1)
			# ax = plt.subplot(groups.ngroups/ncols+1,ncols,i+1)
			if useAx is None:
				ax = plt.subplot(nrows,ncols,i+1)
			else:
				ax = useAx
			temp_key = groups.get_group(k)	

			temp_data = od.ix[:,temp_key.index.values]

			if not fixed_colors:
				categories = tuple(temp_key[colorby].unique())
				color_lookup = dict(zip(categories,sns.color_palette("hls", len(categories))))

			if colorby:
				color_usage = dict(zip(categories,[False]*len(categories)))

			#print temp_key.Well,temp_data.head()

			if fixed_y:
				ylim=(min(od.min()),max(od.max()))
			else:
				ylim = (min(temp_data.min()),max(temp_data.max()))

			if colorby:
				for well,temp_od in temp_data.iteritems():

					#remove na's
					select = ~temp_od.isnull()
					temp_od = temp_od[select]
					temp_time = time[select]

					category = temp_key[colorby][temp_key.index.values==well].values[0]

					if not color_usage[category]:
						# plot using a label, set the ylimits
						plt.plot(temp_time,temp_od.values,c=color_lookup[category],label=str(category))
						plt.ylim(ylim)
						color_usage[category] = True
					else:
						# plot without the label
						plt.plot(temp_time	,temp_od.values,c=color_lookup[category])
				if len(categories) <= max_legend:
					plt.legend(loc="best",**legend_kwargs)
			else:
				for well,temp_od in temp_data.iteritems():

					#remove na's
					select = ~temp_od.isnull()
					temp_od = temp_od[select]
					temp_time = time[select]

					plt.plot(temp_time,temp_od.values)
					plt.ylim(ylim)

			if len(condition)==1:
				k = [k]
			ax.set_title(" ".join(str(k[j]) for j in title_index))

		plt.tight_layout()

		if output == "" and newFig:
			plt.show()
		elif output != "":
			plt.savefig(output)
			plt.close()

	def format(self,fmt,**kwargs):
		if fmt == "standard":
			return self.key,self.data
		elif fmt == "gp" or fmt == "regression":
			return self._expand_data(**kwargs)
		elif fmt == "stacked":
			return self._stack_data(**kwargs)
		else:
			raise NotImplemented("unsupported format")		

	def getData(self,format="standard",**kwargs):
		# if format == "standard":
		# 	return self.key,self.data
		# elif format == "gp":
		# 	return self._expand_data(**kwargs)
		# else:
		# 	pass
		return self.format(format,**kwargs)

	def gp_fit(self,input_cols=[],ard=True,thinning=0):
		import GPy

		edata = self.getData(format="gp",thinning=thinning)

		if not "time" in input_cols:
			input_cols.insert(0,"time")

		x = edata[input_cols]
		y =  edata.od.values[:,np.newaxis]
		self.k = GPy.kern.RBF(x.shape[1],ARD=ard)
		self.m = GPy.models.GPRegression(x,y,self.k)
		self.m.optimize()

	def _expand_data(self,thinning=0):

		# after adding metadata, this is the index time will be sitting at
		time_ind  = self.key.shape[1]

		temp = self.data.iloc[:,1:]
		temp.index = self.data.time
		temp.columns = temp.columns.astype(np.int64)

		combine = pd.merge(self.key,temp.T,left_index=True,right_index=True)

		# subtract blank values
		# blank = combine[combine.strain=="blank"].iloc[:,time_ind:].mean()
		# combine.iloc[:,time_ind:] = combine.iloc[:,time_ind:] - blank

		# expand rows
		# combine = combine[combine.strain!="blank"]
		r = combine.iloc[0,:]
		expand_data = _expand_data_row(r.iloc[time_ind:],thinning)
		for c in r.index[:time_ind]:
				expand_data[c] = r[c]

		for i in range(1,combine.shape[0]):
			r = combine.iloc[i,:]
			temp = _expand_data_row(r.iloc[time_ind:],thinning)
			for c in r.index[:time_ind]:
				temp[c] = r[c]
			expand_data = expand_data.append(temp)

		# remove blank rows
		# expand_data = expand_data[expand_data['strain'] != 'blank']

		# Booleanize variables
		#expand_data = _metadata_cartesian_product(expand_data,columns=["strain"],removeOld=True)
		#expand_data = _metadata_cartesian_product(expand_data,columns=["batch"],removeOld=True)

		#remove NaN timepoints
		expand_data = expand_data[~expand_data.od.isnull()]

		if "BiologicalReplicate" in expand_data.columns:
			expand_data = _metadata_cartesian_product(expand_data,columns=["BiologicalReplicate"],prefix=True,removeOld=True)
		if "TechnicalReplicate" in expand_data.columns:
			expand_data = _metadata_cartesian_product(expand_data,columns=["TechnicalReplicate"],prefix=True,removeOld=True)

		return expand_data

	def _stack_data(self,meta=None,groupby=None,thinning=None):

		reg = self._expand_data(thinning=thinning)

		if not groupby is None:
			g = self.key.groupby(groupby)

			x = []
			y = []

			for vals,ind in g.groups.iteritems():
				select = reg[groupby] == vals
				temp = reg[select]
				x.append(reg[['time']+meta])
				y.append(reg.od[:,None])

			return x,y

		temp = [self.data.time.tolist()] + [[self.key[m][0]]*self.data.shape[0] for m in meta]
		temp = np.column_stack(temp)
		x = [temp]

		for i in range(1,self.data.shape[1]-1):
			temp = [self.data.time.tolist()] + [[self.key[m][i]]*self.data.shape[0] for m in meta]
			temp = np.column_stack(temp)
			x.append(temp)

		y = []
		for i in range(1,self.data.shape[1]):
			temp = self.data.iloc[:,i]
			select = ~temp.isnull()
			temp = temp[select]
			x[i-1] = x[i-1][np.where(select)[0],:]
			y.append(temp[:,None])

		return x,y

	def filter(self,time,od):
		"""Select samples that reach a specific OD cutoff by a specific time"""
		ind_cutoff = np.where(self.data.time == time)[0]

		select = self.data.iloc[ind_cutoff,1:] > od
		select = select.iloc[0,:]

		self.data = self.data[['time']+self.data.columns[1:][select].tolist()]
		self.key = self.key.iloc[self.data.columns[1:],:]

	def select(self,**kwargs):
		"""Filter data by experimental designs."""

		data_copy = self.data.copy()
		key_copy = self.key.copy()

		selection = [True]*self.key.shape[0]

		for k,v in kwargs.iteritems():
			if k in key_copy.columns:
				# if v is np.nan:
				# if np.isnan(v):
				checked = False
				try:
					if np.isnan(v):
						selection = np.all((selection,key_copy[k].isnull()),0)
						checked = True
				except TypeError,e:
					pass
				if not checked:
					selection = np.all((selection,key_copy[k]==v),0)

		selection = np.where(selection)[0]
		key_copy = key_copy.loc[selection,:]
		data_copy = data_copy.loc[:,['time']+(selection).tolist()]

		return GrowthData(data_copy,key_copy,self.logged)

	def applyToData(self,f):
		self.data.iloc[:,1:] = f(self.data.iloc[:,1:])

	def transform(self,subtract=None,scale=None,log=None):
		"""Apply various transformations to the OD data."""
		if not subtract is None:
			self.subtract(subtract)
		if scale:
			self.scale()
		if log and not self.logged:
			if type(log) is bool:
				self.applyToData(np.log2)
			self.logged = True

	def subtract(self,ind):
		"""Subtract the OD of a given time point."""

		# subtract
		if self.logged:
			self.data.iloc[:,1:] = self.data.iloc[:,1:] - self.data.iloc[ind,1:]
		# divide
		else:
			self.data.iloc[:,1:] = self.data.iloc[:,1:]/self.data.iloc[ind,1:]

	def scale(self):

		self.data = self.data.iloc[:,1:] = \
			(self.data.iloc[:,1:] - np.mean(self.data.iloc[:,1:],1))/ np.std(self.data.iloc[:,1:],1)

	def poly_scale(self,p,ind=None,groupby=None):
		"""Scale growth data by a polynomial of degree p, using the first ind datapoints, grouping by groupby."""
		if ind == None:
			ind = 5
		if groupby is None:
			group = {(None,self.key.index)}
		else:
			group = self.key.groupby(groupby)

		time = self.data.time.iloc[:ind]

		for k,index in group.groups.iteritems():
		    temp = self.data.loc[:,index]
		    od = temp.values[:ind,:].ravel()
		    
		    coeff = np.polyfit(time.tolist()*temp.shape[1],od,p)
		    
		    temp = temp - np.polyval(coeff,self.data.time.values[0])
		    self.data.loc[:,index] = temp

def _parse_time(t):
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

def _expand_data_row(r,thinning=0):
	well = int(r.name)
	r = pd.DataFrame(r)
	r.columns = ["od"]
	r['Well'] = well
	r['time'] = r.index

	if thinning > 0:
		select = np.arange(0,r.shape[0],thinning)
		r = r.iloc[select,:]

	return r

# given an array X and columns,
# create cartesian product of all combinations of values in each column
# If hierarchy is true, follows the hierarchy implied by the column order
# and higher level columns get a seperate output column
def _metadata_cartesian_product(X,columns,hierarchy=False,prefix=False,removeOld=False):

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