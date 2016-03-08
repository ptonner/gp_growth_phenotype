import pandas as pd
import numpy as np
from ConfigParser import ConfigParser
from registry import registry
from gp_growth import storage
import os, gc

class Analysis():

	def __init__(self,path):
		self.loaded = False
		self.path = path

		assert "config.txt" in os.listdir(self.path),"Must provide config.txt file for analysis!"

		self.configFile = os.path.join(self.path,"config.txt")

		self.load()

	def load(self):
		"""load configuration of analysis from the config.txt file in self.path
		"""

		# read in the config file
		cfile = self.configFile
		configParse = ConfigParser()
		configParse.read(cfile)

		# location of data file (h5, sql, etc.)
		storage.open(configParse.get("analysis","data"))

		# what search terms do we want to use for the data?
		self.kwargs = dict(configParse.items("data"))
		for k in self.kwargs.keys():
			self.kwargs[k] = self.kwargs[k].split(",")
		# get data matching search terms
		self.data = storage.getData(**self.kwargs)

		# determine all factories to use in analysis
		self.factories = []
		for f in configParse.get("analysis","factory").split(","):
			self.factories.append((f,registry[f]))
		# self.factory = registry[configParse.get("analysis","factory")]

		# how to group data samples
		self.groupby = configParse.get("analysis","groupby").split(",")

		# what is the max number of curves per grouping?
		self.maxSize = int(configParse.get("analysis","maxSize"))

		# everything is loaded
		self.loaded = True

	def run(self,verbosity=None):
		if verbosity is None:
			verbosity = 0

		if not self.loaded:
			self.load()

		g = self.data.key.groupby(self.groupby)

		for name,f in self.factories:

			fact = f()

			paramFile = name+"_params.csv"
			if paramFile in os.listdir(self.path):
				fact.savedParameters = pd.read_csv(os.path.join(self.path,name+"_params.csv"))
			paramFile = os.path.join(self.path,name+"_params.csv")

			#iterate over unique groupings and fit the model from factory
			for vals in g.groups.keys():
				if len(self.groupby) == 1:
					vals = [vals]

				# check for data we've already fit a model for
				skip = False
				if fact.savedParameters.shape[0] > 0:
					params = fact.savedParameters[self.groupby]
					for i in range(params.shape[0]):
						for j in range(len(vals)):

							# print params.iloc[i,j], vals[j]

							# if type(vals[j]) == float and type(params.iloc[i,j]) == float:
							# if type(vals[j]) == float:
							# if np.isnan(vals[j]) or params.isnull().iloc[i,j]:
							try :
								# if not np.isnan(vals[j]) or not np.isnan(params.iloc[i,j]):
								if np.isnan(vals[j]) or params.isnull().iloc[i,j]:
									if not np.isnan(vals[j]) or not params.isnull().iloc[i,j]:
										break
							except:
							# else:
								compare = str(params.iloc[i,j])
								if compare != vals[j]:
									# print "mismatch"
									break

							# increment to test if we reached the end
							j += 1

						if j < len(vals):
							# print "failed"
							continue

						# didn't fail, found a match
						skip = True
				if skip:
					continue	

				# build a dictionary of key-value pairs for selecting data
				kwargs = dict(zip(self.groupby,vals))

				temp = self.data.select(**kwargs)

				if verbosity > 0:
					print kwargs, temp.key.shape[0]

				if temp.key.shape[0] > self.maxSize:
					if verbosity > 0:
						print "Number of samples to large (%d > %d)" % (temp.key.shape[0],self.maxSize)
					continue

				regression = temp.getData("gp")

				gp = fact.build(regression,optimize=True)
				fact.saveOptimizedParameters(gp,**kwargs)
				fact.savedParameters.to_csv(paramFile,index=False)

				# del gp
				# del temp
				collected = gc.collect()
				if verbosity > 0:
					print "Garbage collector: collected %d objects." % (collected)

if __name__ == "__main__":
	import sys, getopt

	# buffer flush
	from lib.unbuffered import Unbuffered
	sys.stdout = Unbuffered(sys.stdout)

	opts,args = getopt.getopt(sys.argv[1:],'v')

	verbosity = 0
	for opt,val in opts:
		if opt == "-v":
			verbosity = 1

	assert len(args) > 0, "must provide directory for analysis!"

	analysis = Analysis(args[0])
	analysis.run(verbosity=verbosity)