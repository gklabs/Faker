import pandas as pd
from collections import defaultdict
from pathlib import Path
import json

class read_data:
	def __init__(self,filenames,path):
		self.df = pd.DataFrame()
		self.path = path
		self.filenames = filenames

	def getdata(self):
		dictionary={}
		count=0

		for name in self.filenames:
			try:
				with  open(self.path+name+"/news content.json") as f:
					data= json.load(f)
					count=count+1
					if data['text'].strip() != '':
						dictionary[name] = [data['url'], data['text']]
			except IOError:
				pass

			uid= list(dictionary.keys())
			values= list(dictionary.values())

			url= [i[0] for i in values]
			text= [i[1] for i in values]

			data_tuples = list(zip(uid,url,text))
			self.df= pd.DataFrame(data_tuples, columns=['uid','url','text'])
			self.df.to_csv("fnndataset.csv")

	

		# for text in df.text:
		# 	templist=tokenizer.encode(text,max_length= len(text))
		# 	print(templist)
		# 	kk.append(templist.tokens)
		# df.text = kk
