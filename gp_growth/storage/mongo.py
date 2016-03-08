from pymongo import MongoClient
import pandas as pd
from ..data.growth import GrowthData

class MongoDB():

	def __init__(self,):
		self.client = MongoClient()
		self.db = self.client.db

		self.projects = self.db.projects
		self.plates = self.db.plates
		self.wells = self.db.wells
		self.data = self.db.data
		self.experimentalDesigns = self.db.experimentalDesigns

	def clean(self):
		self.projects.drop()
		self.plates.drop()
		self.wells.drop()
		self.data.drop()
		self.experimentalDesigns.drop()

	def create(self,project,plate,data,key,meta_types={},useDataColumns=False):

		size = key.shape[0]

		project = self.getProject(project,create=True)
		plate = self.createPlate(plate,size,project)

		if useDataColumns:
			self.createWells(plate,data.columns)
			self.createData(plate,data,data.columns)
			self.createMetadata(plate,key,data.columns)
		else:
			self.createWells(plate)
			self.createData(plate,data)
			self.createMetadata(plate,key,)

	def createProject(self,name):
		project = self.projects.insert_one({"name":name})
		return project.inserted_id

	def createPlate(self,name,size,project):
		plate = self.plates.insert_one({"name":name,"size":size,'project':project})
		return plate.inserted_id

	def createWells(self,plate,labels=None):

		size = self.plates.find_one(plate)['size']

		for i in range(size):
			well = {"name":i,"plate":plate}
			if not labels is None:
				well['name'] = labels[i]

			self.wells.insert_one(well)

	def createData(self,plate,data,labels=None):

		time = data.values[:,0].tolist()
		size = self.plates.find_one(plate)['size']

		for i in range(size):

			ind = i+1
			l = i
			if not labels is None:
				l = labels[i]
			well = self.wells.find_one({"plate":plate,"name":l})['_id']
			od = data.values[:,ind].tolist()
			timeseries = {"well":well,"time":time,"od":od}
			self.data.insert_one(timeseries)

	def createMetadata(self,plate,meta,labels=None,types={}):

		for i in range(meta.shape[0]):
			l = i
			if not labels is None:
				l = labels[i]

			ind = meta.index[i]

			for name in meta.columns:
				# if name == "Well":
				# 	continue
				well = self.wells.find_one_and_update({"plate":plate,"name":l},{"$set":{name:meta.loc[ind,name]}})

	def getProject(self,name,create=False):
		project = self.projects.find_one({"name":name})

		if project is None:
			if create:
				self.createProject(name)
			else:
				raise ValueError("no project names %s!" % name)

		return project

	def getPlates(self,):
		return [x['name'] for x in self.plates.find()]

	def getWells(self,**kwargs):
		if "plate" in kwargs:
			if type(kwargs['plate']) == list:
				kwargs['plate'] = {"$in":[x["_id"] for x in self.plates.find({"name":{"$in":kwargs['plate']}})]}
			else:
				kwargs['plate'] = self.plates.find_one({"name":kwargs['plate']})["_id"]

		for k in kwargs.keys():
			if k == "plate":
				continue
			if type(kwargs[k]) == list:
				kwargs[k] = {"$in":kwargs[k]}

		wells = self.wells.find(kwargs)
		wells = [w['_id'] for w in list(wells)]
		return wells

	def getData(self,logged=None,subtract=None,**kwargs):

		wells = self.getWells(**kwargs)

		data = self.data.find({"well":{"$in":wells}})

		# od = None
		# ind = 0
		# for d in data:
		# 	# label = self.wells.find_one(d['well'])['name']
		# 	label = ind
		# 	temp = pd.DataFrame({"time":d['time'],label:d['od']})

		# 	if od is None:
		# 		od = temp
		# 	else:
		# 		od = pd.merge(od,temp,how="outer",on="time")

		# 	ind += 1

		od = None
		meta = None
		ind = 0
		for w in wells:
			d = self.data.find_one({"well":w})
			w = self.wells.find_one(w)

			temp = pd.DataFrame({"time":d['time'],ind:d['od']})

			if od is None:
				od = temp
			else:
				od = pd.merge(od,temp,how="outer",on="time")

			del w['_id']
			w['plate'] = self.plates.find_one(w['plate'])['name']
			temp = pd.DataFrame(w,index=[ind])
			temp['well'] = ind
			del temp['name']

			if meta is None:
				meta = temp
			else:
				meta = pd.concat((meta,temp))

			ind += 1

		cols = []
		for c in od.columns:
			if not c == "time":
				cols.append(c)
		od = od[['time']+cols]

		ret = GrowthData(od,meta)

		if not logged is None or subtract is None:
			ret.transform(log=logged,subtract=subtract)

		return ret

	def getExperimentalDesigns(self,name,**kwargs):
		wells = self.getWells(**kwargs)

		ret = {}

		for w in wells:
			w = self.wells.find_one(w)
			if name in w:
				if not w[name] in ret:
					ret[w[name]] = None

		return ret.keys()

