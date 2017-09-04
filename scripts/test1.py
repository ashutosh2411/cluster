
from __future__ import print_function
from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import numpy as np

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space', 'rec.sport.hockey', 'rec.motorcycles']
print("Loading 20 newsgroups dataset for categories:")
print(categories)

dataset = sklearn.datasets.load_files('/home/mabrin/project/cluster/datasets/20news-bydate/20news-bydate-train', description=None, categories=categories , load_content=True, shuffle=True, encoding = 'utf-8',decode_error='ignore', random_state=42)


print("%d documents" % len(dataset.data))
print("%d categories" % len(dataset.target_names))
print()
print(dataset.target)
true_k = 8

vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                 min_df=4, stop_words='english',
                                 use_idf=True)
X = vectorizer.fit_transform(dataset.data[0:64])

km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000)
km.fit(X)

print()

clusters = defaultdict(list)

k = 0;
for i in km.labels_ :
  clusters[i].append(dataset.filenames[k])  
  k += 1
  
for clust in clusters :
  print ("\n************************\n")
  for filename in clusters[clust] :
    print (filename)
       
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :15]:
            print(' %s' % terms[ind], end='')
        print()
        
