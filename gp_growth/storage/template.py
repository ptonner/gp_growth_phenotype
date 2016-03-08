from copy import copy

def replaceTemplateStrings(rep,**kwargs):
    rep = copy(rep)
    for k in rep.keys():
        v = rep[k]

        if isinstance(v,list):
            temp = []
            for i in range(len(v)):
                if isinstance(v[i],str):
                    if v[i][0] == "%":
                        temp2 = v[i][1:]
                        if temp2 in kwargs:
                            temp.append(kwargs[temp2])
                            continue
                #         else:
                #             temp.append(v[i])
                #     else:
                #         temp.append(v[i]
                # else:
                temp.append(v[i])
            rep[k] = temp        
        elif isinstance(v,str) and v[0] == "%":
            temp = v[1:]
            if temp in kwargs:
                rep[k] = kwargs[temp]
                
    return rep

class Template(object):
	"""A template class for searching the database"""

	def __init__(self,search_template):
		self.searchTemplate = search_template

	def convert(self,**kwargs):
		return replaceTemplateStrings(self.searchTemplate,**kwargs)