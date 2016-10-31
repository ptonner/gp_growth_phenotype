import os,re,sys
import getopt
import numpy as np
from gp_growth import utils
import pandas as pd

hsal_dir = "data/raw/hsal_ko"
plates = os.listdir(hsal_dir)

meta = pd.DataFrame()
od = pd.DataFrame()

for plate in plates:

	print plate

	try:
	    key,data = utils.load_bioscreen(os.path.join(hsal_dir,plate),removeBlank=False)
	except ValueError, e:
	    key,data = utils.load_bioscreen(os.path.join(hsal_dir,plate),convert=True,removeBlank=False)

	print key.shape

	if meta.shape[0] > 0:
		meta = pd.concat((meta,key))
		od = pd.merge(od,data,on="Time")
	else:
		meta = key
		od = data

	print meta.shape,od.shape

meta.index=range(meta.shape[0])
od.columns=['time']+meta.index.tolist()

combined = pd.merge(meta,od.T,left_index=True,right_index=True)
combined.columns = combined.columns[:meta.shape[1]].tolist() + od.time.tolist()
#combined = combined[combined.Strain.isin(['ura3','rosR','trmB','asnC','idr1','idr2','sirR','VNG1179'])]
#combined = combined[combined['mM PQ'].isin([0,0.333])]

meta.to_csv("data/processed/meta.csv",index=False)
od.to_csv("data/processed/data.csv",index=False)
combined.to_csv("data/processed/combined.csv",index=False)
