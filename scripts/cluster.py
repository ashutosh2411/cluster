from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import sklearn

categories = [
   "business" ,"sport" , 'entertainment', 'tech','politics']

dataset = sklearn.datasets.load_files('/home/mabrin/project/cluster/datasets/bbcTest/',
                                      description=None, categories=categories ,
                                      load_content=True, shuffle=True, 
                                      encoding = 'utf-8',decode_error='ignore',
                                      random_state=25)


true_k = 5 #no of groups

vectorizer = TfidfVectorizer(max_df=.3, max_features=175 ,min_df=10 ,
                                    stop_words='english',use_idf=True)
                                    
X = vectorizer.fit_transform(dataset.data[:125])

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                    init_size=1000,batch_size=1000)
km.fit(X)

Y = vectorizer.fit_transform(dataset.data[125:])

cur = Y.toarray()

clusters = defaultdict(list)

k = 0
for i in km.labels_ :
  clusters[i].append(dataset.filenames[k])  
  k += 1

left_doc , tmp = Y.shape

tmp = [] 
for i in range(0,left_doc):
    clust = []
    for j in range (5):
        x = km.cluster_centers_[j]
        y = cur[i]
        clust.append(sklearn.metrics.pairwise.euclidean_distances(x.reshape(1,-1) ,
                 y.reshape(1,-1), Y_norm_squared=None, squared=False, X_norm_squared=None))
    #print(np.argmin(clust))
    tmp=tmp+[i]

outfile = "out.txt"
np.savetxt(outfile,(np.hstack((dataset.filenames[125:],tmp))) ,delimiter = ',')   

