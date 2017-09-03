from os import listdir
from os.path import isfile, join
import sklearn.feature_extraction.text as extract

#print len(files)
vect = extract.TfidfVectorizer( analyzer = 'word', 
								max_df=.4, max_features=5,
                                min_df=.05, stop_words='english',
                                decode_error='ignore', 
                                norm = 'l2')

files = [f for f in listdir('../datasets/bbc/business/') if isfile(join('../datasets/bbc/business/', f))]
for x in files:
	print x
	file_name = '../datasets/bbc/business/'+x
	file = open(file_name)
	file_data = file.read()
	X = vect.fit(open (file_name))
	idf = vect.idf_
	print (vect.get_feature_names(), idf, vect.tf_)
"""
files = [f for f in listdir('../datasets/bbc/sport/') if isfile(join('../datasets/bbc/sport/', f))]
for x in files:
	print x
	file_name = '../datasets/bbc/sport/'+x
	file = open(file_name)
	file_data = file.read()
	X = vect.fit(open (file_name))
	idf = vect.idf_
	print dict(zip(vect.get_feature_names(), idf))

files = [f for f in listdir('../datasets/bbc/politics/') if isfile(join('../datasets/bbc/politics/', f))]
for x in files:
	print x
	file_name = '../datasets/bbc/politics/'+x
	file = open(file_name)
	file_data = file.read()
	X = vect.fit(open (file_name))
	idf = vect.idf_
	tf  = vect.tf_
	print (vect.get_feature_names(),idf,tf)

"""