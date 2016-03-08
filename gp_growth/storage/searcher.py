import template

class Searcher(template.Template):
	"""A utility class for defining a templated search and preprocessing of data."""

	def __init__(self,searchTemplate,db):
		template.Template.__init__(self,searchTemplate)
		self.db = db
	
	def search(self,verbose=False,**kwargs):
		searchKwargs = self.convert(**kwargs)

		# add kwargs not in template
		for k in kwargs.keys():
			if not k in self.searchTemplate:
				searchKwargs[k] = kwargs[k]

		if verbose:
			print "Searching on",searchKwargs

		data = self.db.getData(**searchKwargs)
		data = self.process(data)
		return data

	def __call__(self,**kwargs):
		return self.search(**kwargs)

	def process(self,data):
		"""Override this method to apply a transformation to the returned data"""
		return data