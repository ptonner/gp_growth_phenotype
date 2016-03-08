import growth

class DataFilter(object):
    
    def __call__(self,data,copy=True,**kwargs):
        return self.filter(data,**kwargs)
    
    def filter(self,data,copy=True,**kwargs):
        if copy:
            data = growth.GrowthData(data.data.copy(),data.key.copy(),logged=True)
        return self._filter(data,**kwargs)
    
    def _filter(self,data,**kwargs):
        raise NotImplemented("must implement filter method!")