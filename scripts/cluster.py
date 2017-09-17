"""
Program to cluster large number of documents
"""
import resource 
import collections
from os import listdir
from os.path import isfile, join
from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import sklearn

def using(point = ""):
	usage = resource.getrusage(resource.RUSAGE_SELF)
	return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )
u_start = using('start')
data = '../datasets/bbc'
categories = [f for f in listdir(data) if not isfile(join(data, f))]
#print categories
#categories = ["business" ,"sport", 'entertainment', 'tech', 'politics']

dataset = sklearn.datasets.load_files(	data + '/',	description=None, categories=categories ,
										load_content=True, shuffle=True, 
										encoding = 'utf-8',decode_error='ignore',
										random_state=179863)
true_k = len(categories) 	#no of groups
u_dataRead = using()
vectorizer = TfidfVectorizer(max_df=.5, max_features=310 ,min_df=10 ,
										stop_words='english',use_idf=True)				
dat = vectorizer.fit_transform(dataset.data)
dat = dat.toarray()
u_Vect = using()

factor = int(len(dataset.data) / 20)

X = dat[:factor]
Y = dat[factor:]

km = KMeans(n_clusters=true_k)
km.fit(X)

cur = Y
clusters = defaultdict(list)
vishal = defaultdict(list)
k = 0
for i in km.labels_ :
	clusters[i].append(dataset.filenames[k]) 
	vishal[i].append(dataset.filenames[k][:-7]) 
	k += 1
u_afterTrain = using()
left_doc = Y.shape[0]

tmp = [] 

for i in range(0,left_doc):
	clust = []
	for j in range (true_k):
		x = km.cluster_centers_[j]
		y = cur[i]
		clust.append(sklearn.metrics.pairwise.euclidean_distances(x.reshape(1,-1) ,
				 y.reshape(1,-1), Y_norm_squared=None, squared=False, X_norm_squared=None))

	k =np.argmin(clust)
	clusters[k].append(dataset.filenames[i+factor])
	vishal[k].append(dataset.filenames[i+factor][:-7]) 	
#	print (dataset.filenames[i+factor])
u_afterCluster = using()

#for i in range(true_k):
#		for j in clusters[i]:
#			print(j)
#		print ""
sum = 0
#print len(dataset.data)
#print type(vishal[i])
for i in range (len(vishal)):
	counter = collections.Counter(vishal[i])
	sum = sum + len(vishal[i]) - max(counter.values())
print 'accuracy: ',(1-float(sum)/len(dataset.data))*100
print (u_afterCluster)
