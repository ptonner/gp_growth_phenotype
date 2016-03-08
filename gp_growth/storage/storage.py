#from tables import *
from numpy import *
import pandas as pd
import os,sys
from ..data.growth import GrowthData
from datatypes import *

def open(fname,overwrite=False,warn=True):
	global h5file
	
	if os.path.isfile(fname) and not overwrite:
		h5file = open_file(fname, mode = "r+", title = "Growth data")
	else:
		if os.path.isfile(fname) and warn:
			print fname,"already exists, overwrite?"
			select = raw_input()
			if select.lower() != "y":
				return
		h5file = open_file(fname, mode = "w", title = "Growth data")

def close():
	h5file.close()	

###################################################################
## Data input functions
###################################################################

def create(plate,data,key,plate_type=PLATE_TYPES[0],meta_types={}):
	plate = createPlate(plate,key.shape[0])
	createData(plate,data,plate_type)
	createMetadata(plate,key,meta_types)

def createPlate(name,size):
	name = name.replace(" ","")
	plate = h5file.create_group("/",name,"")
	h5file.create_array(plate,'size',size)
	return plate

def createData(plate,data,plate_type=PLATE_TYPES[0]):
	if plate_type == "Bioscreen":
		table = h5file.create_table(plate,"data",Bioscreen)
	elif plate_type == "Plate96Well":
		table = h5file.create_table(plate,"data",Plate96Well)
	elif plate_type == "Plate384Well":
		table = h5file.create_table(plate,"data",Plate384Well)
	else:
		# not implemented yet
		raise NotImplementedError("No implmentation for %s" % plate_type) 

	row = table.row
	for i in xrange(data.shape[0]):
		row['time'] = data.iloc[i,0]
		for j in xrange(1,data.shape[1]):
			row['well%d'%(j-1)] = data.iloc[i,j]
		row.append()

	table.flush()

def createMetadata(plate,meta,types={}):
	meta_ind = 0

	meta_table = h5file.create_table(plate,"meta",Metadata,"Metadata")
	meta_row = meta_table.row

	well_index = arange(meta.shape[0])

	for col in meta:
		if col == "Well":
			continue
		vals = meta[col].unique()
		for v in vals:
			# wells = meta['Well'][meta[col] == v]
			wells = well_index[where(meta[col] == v)[0]]

			meta_name = "wells_%d" % meta_ind; meta_ind += 1
			h5file.create_array(plate,meta_name,wells)
			
			meta_row['name'] = col
			meta_row['val'] = v
			if col in types:
				meta_row['type'] = types[col]

			if meta_row['type'] == METADATA_TYPES.index("str"):
				meta_row['val'] = str(meta_row['val'])
			elif meta_row['type'] == METADATA_TYPES.index("int"):
				meta_row['val'] = int(meta_row['val'])
			elif meta_row['type'] == METADATA_TYPES.index("float"):
				try:
					meta_row['val'] = float(meta_row['val'])
				except ValueError, e:
					meta_row['val'] = nan
			# else:
			# 	meta_row['type'] = METADATA_TYPES[0]

			meta_row['well_array'] = meta_name
			meta_row.append()

	meta_table.flush()

###################################################################
## Data output functions
###################################################################

def getPlates():
	return [p._v_name for p in h5file.root]

def getExperimentalDesigns(designName,**kwargs):

	w = getWells(**kwargs)
	ed = _get_experimentalDesign(w)

	if designName in ed.columns:
		return ed[designName].unique()
	else:
		return []

	# vals = {}

	# for p in h5file.root:
	# 	plate_designs = p.meta
	# 	for ed in plate_designs:
	# 		if ed['name'] == designName:
	# 			#search = experimental_designs[ed['name']]

	# 			# if (ed['val'] == search) or (type(search) == list and (ed['val'] in search)):

	# 			# 	w_temp = p.__getattr__(ed['well_array'])
	# 			# 	w_temp = [a for a in w_temp.iterrows()]
	# 			# 	w_temp = set(w_temp)

	# 			# 	# if type(search) == list:
	# 			# 	# 	w = w.union(w_temp)
	# 			# 	# else:
	# 			# 	# 	w = w.intersection(w_temp)

	# 			# 	if len(hits[ed['name']]) == 0:
	# 			# 		hits[ed['name']] = w_temp
	# 			# 	else:
	# 			# 		hits[ed['name']] = hits[ed['name']].union(w_temp)
	# 			if not ed['val'] in vals:
	# 				vals[ed['val']] = None

	# return vals.keys()


def getWells(plates=None,verbose=False,**kwargs):

	if plates is None:
		plates = []

	wells = []

	wellSelect = None
	if 'well' in kwargs:
		wellSelect = kwargs['well']
		del kwargs['well']

	experimental_designs = kwargs

	if len(plates) == 0:
		for plate in h5file.root:
			plates.append(plate)
	else:
		temp = []
		for plate in h5file.root:
			if plate._v_name in plates:
				temp.append(plate)
		plates = temp
	
	count = 0
	for p in plates:
		# num_wells = len(p.data.colnames) - 1
		num_wells = list(p.size)[0]
		w = set([i for i in range(num_wells)])

		hits = dict([(k,[]) for k in experimental_designs.keys()])

		if not wellSelect is None:
			if type(wellSelect) == ndarray:
					wellSelect = wellSelect.tolist()
			if type(wellSelect) != list:
				wellSelect = [wellSelect]
			hits['well'] = set(wellSelect)

		plate_designs = p.meta
		for ed in plate_designs:
			if ed['name'] in experimental_designs:
				search = experimental_designs[ed['name']]
				if not type(search) == str:
					search = str(search)

				if type(search) == ndarray:
					search = search.tolist()

				if (ed['val'] == search) or (type(search) == list and (ed['val'] in search)):

					w_temp = p.__getattr__(ed['well_array'])
					w_temp = [a for a in w_temp.iterrows()]
					w_temp = set(w_temp)

					# if type(search) == list:
					# 	w = w.union(w_temp)
					# else:
					# 	w = w.intersection(w_temp)

					if len(hits[ed['name']]) == 0:
						hits[ed['name']] = w_temp
					else:
						hits[ed['name']] = hits[ed['name']].union(w_temp)

		# did we hit everything?
		success = all([len(v)>0 for k,v in hits.iteritems()])

		# add the intersection of all wells
		if success:
			keys = hits.keys()
			if len(keys) > 0:
				w = hits[keys[0]]
				for k in keys[1:]:
					w = w.intersection(hits[k])

				w = list(w)
				wells.append((p,w))
			else:
				wells.append((p,list(w)))

		count += 1

		if verbose:
			perc = (100.*count)/len(plates)
			print "\rFinding wells...%.2f%% (%d wells)"% (perc,sum([len(x) for y,x in wells])),
			sys.stdout.flush()

	return wells


# def getData(plates=[],experimental_designs={}):
def getData(plates=None,verbose=False,logged=None,subtract=None,*args,**kwargs):

	if logged is None:
		logged = True
	if subtract is None:
		subtract = True

	wells = getWells(plates,verbose,**kwargs)

	if verbose:
		print ""

	# return _get_data(wells)
	data = _generate_data(wells,verbose)

	if logged:
		data.transform(log=True)
	if subtract:
		data.transform(subtract=0)

	return data

def _generate_data(wells,verbose=False):
	
	key,data = _get_data(wells,verbose)

	return GrowthData(data=data,key=key)

def _get_experimentalDesign(wells,verbose=False):
	key = pd.DataFrame()
	count = 0
	ind = 0
	for plate,w_ind in wells:

		designs = []
		for row in plate.meta:
			if not row['name'] in designs:
				designs.append(row['name'])

		if len(w_ind) == 0:
			count+= 1
			continue

		temp = pd.DataFrame([[""]*(1+len(designs))]*len(w_ind),columns=['well']+designs)

		for row in plate.meta:
			array_indexes = [i for i in plate.__getattr__(row['well_array'])]
			search = [w in array_indexes for w in w_ind]
			if any(search):
				col_index = where(temp.columns==row['name'])[0]

				temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index] = [row['val']]

				if row['type'] != METADATA_TYPES.index("str"):
					if row['type'] == METADATA_TYPES.index("int"):
						temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index] = temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index].astype(int)
					elif row['type'] == METADATA_TYPES.index("float"):
						temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index] = temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index].astype(float)

		new_indexes = range(ind,ind+len(w_ind))
		temp['well'] = new_indexes
		temp['batch'] = plate._v_name

		if key.shape[0] == 0:
			key = temp
		else:
			key = key.append(temp)

		
		ind += len(w_ind)
		count += 1

		if verbose:
			perc = (100.*count)/len(wells)
			print "\rBuilding meta table...%.2f%%"%perc,
			sys.stdout.flush()

	return key

def _get_data(wells,verbose=False):

	key = pd.DataFrame()
	data = pd.DataFrame()

	# build tables
	count = 0
	ind = 0
	for plate,w_ind in wells:

		new_ids = [i for i in range(ind,ind+len(w_ind))]
		temp = pd.DataFrame(columns=['time']+new_ids)

		for row in plate.data:
			new = pd.DataFrame(columns=['time']+new_ids,index=[count])
			new['time'] = [row['time']]

			for i,w in enumerate(w_ind):
				well_name = "well%d" % w
				new[ind+i] = [row[well_name]]

			temp = temp.append(new,ignore_index=True)

		if data.shape[0] == 0:
			data = temp
		else:
			# easy update, new and old data have the same time values
			if data.shape[0] == temp.shape[0] and all(data.time==temp.time):
				data = pd.concat((data,temp.iloc[:,1:]),1)
			#ouch
			else:
				data = pd.merge(data,temp,"outer",data.columns[0])

		ind += len(w_ind)
		count += 1

		if verbose:
			perc = (100.*count)/len(wells)
			print "\rBuilding data table...%.2f%%"%perc,
			sys.stdout.flush()

		# print plate._v_name,data.shape,temp.shape

	if verbose:
		print ""

	key = _get_experimentalDesign(wells,verbose)
	# count = 0
	# ind = 0
	# for plate,w_ind in wells:

	# 	designs = []
	# 	for row in plate.meta:
	# 		if not row['name'] in designs:
	# 			designs.append(row['name'])

	# 	if len(w_ind) == 0:
	# 		count+= 1
	# 		continue

	# 	temp = pd.DataFrame([[""]*(1+len(designs))]*len(w_ind),columns=['well']+designs)

	# 	for row in plate.meta:
	# 		array_indexes = [i for i in plate.__getattr__(row['well_array'])]
	# 		search = [w in array_indexes for w in w_ind]
	# 		if any(search):
	# 			col_index = where(temp.columns==row['name'])[0]

	# 			temp.iloc[where([w in array_indexes for w in w_ind])[0],col_index] = [row['val']]

	# 	new_indexes = range(ind,ind+len(w_ind))
	# 	# temp['well'] = new_indexes
	# 	temp['well'] = new_indexes
	# 	temp['batch'] = plate._v_name

	# 	if key.shape[0] == 0:
	# 		key = temp
	# 	else:
	# 		key = key.append(temp)

		
	# 	ind += len(w_ind)
	# 	count += 1

	# 	if verbose:
	# 		perc = (100.*count)/len(wells)
	# 		print "\rBuilding meta table...%.2f%%"%perc,
	# 		sys.stdout.flush()

	# reorder data columns
	# cols = data.columns
	# cols = cols.drop("time")
	# cols = ['time'] + cols.tolist()
	# data = data[cols]

	# reorder key columns
	# cols = key.columns
	# cols = cols.drop("Well")
	# cols = ['Well'] + cols.tolist()
	# key = key[cols]

	key.index = data.columns[1:]

	return key,data
