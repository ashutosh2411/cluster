"""
Program to cluster large number of documents
"""

from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import sklearn

categories = [
   "business" ,"sport", 'entertainment', 'tech', 'politics']

dataset = sklearn.datasets.load_files(	'../datasets/bbcTest/',
										description=None, categories=categories ,
										load_content=True, shuffle=True, 
										encoding = 'utf-8',decode_error='ignore',
										random_state=179863)

true_k = 5 							#no of groups

vectorizer = TfidfVectorizer(max_df=.5, max_features=310 ,min_df=10 ,
										stop_words='english',use_idf=True)
									
dat = vectorizer.fit_transform(dataset.data)
dat = dat.toarray()

X = dat[:125]
Y = dat[125:]

km = KMeans(n_clusters=true_k)
km.fit(X)

cur = Y
clusters = defaultdict(list)

k = 0
for i in km.labels_ :
	clusters[i].append(dataset.filenames[k])  
	k += 1

left_doc = Y.shape[0]

tmp = [] 

for i in range(0,left_doc):
	clust = []
	for j in range (5):
		x = km.cluster_centers_[j]
		y = cur[i]
		clust.append(sklearn.metrics.pairwise.euclidean_distances(x.reshape(1,-1) ,
				 y.reshape(1,-1), Y_norm_squared=None, squared=False, X_norm_squared=None))
	k =np.argmin(clust)
	clusters[k].append(dataset.filenames[i+125])
	
for i in range(5):
		for j in clusters[i]:
			print(j)    
		print()