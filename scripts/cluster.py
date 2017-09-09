from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import sklearn

categories = [
   "business" ,"sport", 'entertainment', 'tech', 'politics']

dataset = sklearn.datasets.load_files('/home/mabrin/project/cluster/datasets/bbcTest/',
                                      description=None, categories=categories ,
                                      load_content=True, shuffle=True, 
                                      encoding = 'utf-8',decode_error='ignore',
                                      random_state=25)


true_k = 5 #no of groups

vectorizer = TfidfVectorizer(max_df=.5, max_features=310 ,min_df=10 ,
                                    stop_words='english',use_idf=True)
                                    
dat = vectorizer.fit_transform(dataset.data)
dat = dat.array()
X = dat[:125]
Y = dat[125:]




km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                    init_size=1000,batch_size=1000)
km.fit(X)



cur = Y

clusters = defaultdict(list)

k = 0
for i in km.labels_ :
  clusters[i].append(dataset.filenames[k])  
  k += 1

left_doc , tmp = Y.shape

tmp = [] 
x = km.cluster_centers_[1]
y = cur[1]
print(np.shape(x.reshape(1,-1)))
print(np.shape(y.reshape(1,-1)))
for i in range(0,left_doc):
    clust = []
    for j in range (5):
        x = km.cluster_centers_[j]
        y = cur[i]
        clust.append(sklearn.metrics.pairwise.euclidean_distances(x.reshape(1,-1) ,
                 y.reshape(1,-1), Y_norm_squared=None, squared=False, X_norm_squared=None))
    k =np.argmin(clust)
    clusters[k].cen
    clusters[k].append(dataset.filenames[i+125])
    
for i in range(5):
        for j in clusters[i]:
                print(j)    
        print()



#outfile = "out.txt"
#np.savetxt(outfile,(np.hstack((dataset.filenames[125:],tmp))) ,delimiter = ',')   

