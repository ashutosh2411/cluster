from collections import defaultdict
import sklearn.datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import sklearn

categories = [
   'business','entertainment',"sport" , 'tech','politics']

dataset = sklearn.datasets.load_files('/home/mabrin/project/cluster/datasets/bbcTest/',
                                      description=None, categories=categories ,
                                      load_content=True, shuffle=False, 
                                      encoding = 'utf-8',decode_error='ignore',
                                      random_state= 70)

datasetT = sklearn.datasets.load_files('/home/mabrin/project/cluster/datasets/bbc/',
                                      description=None, categories=categories ,
                                      load_content=True, shuffle=True, 
                                      encoding = 'utf-8',decode_error='ignore',
                                      random_state= 42)


true_k = 5 #no of groups

vectorizer = TfidfVectorizer(max_df=.6, max_features=250 ,min_df=.01 ,
                                    stop_words='english',use_idf=True)
                                    
X2 = vectorizer.fit_transform(dataset.data[:50]).toarray()
X5 = vectorizer.fit_transform(dataset.data[50:100]).toarray()
X3 = vectorizer.fit_transform(dataset.data[100:150]).toarray()
X4 = vectorizer.fit_transform(dataset.data[150:200]).toarray()
X1 = vectorizer.fit_transform(dataset.data[200:]).toarray()
X6 = vectorizer.fit_transform(datasetT.data).toarray()

X = [X1,X2,X3,X4,X5]


centriole=[]
for i in X:
    print(i.shape)
    s = []
    for k in range(i.shape[1]):
        
        tmp =float(0)
        for l in range(i.shape[0]):
            tmp = tmp+ i[l,k]
        s.append(tmp / 50)
    centriole.append(s)
exit

'''for i in X:
    km = MiniBatchKMeans(n_clusters=1, init='k-means++', n_init=1,
                    init_size=1000,batch_size=1000)
    km.fit(X1)
    centriole.append(km.cluster_centers_[0])
'''

    

cur = X6

clusters = [[],[],[],[],[]]

for i in range(5):
    for filename in dataset.filenames[i*50:(i+1)*50-1]:
        clusters[i].append(filename)

exit

x = centriole[0]
y = cur[0]

   

#centers = km.cluster_centers_
for i in range(0,len(X6)):
    clust = []
    for j in range (true_k):
        x = np.array(centriole[j])
        y = cur[i]
        #print(len(x),type(x))
        #print(len(y),type(y))
        #clust.append(sklearn.metrics.pairwise.euclidean_distances(x.reshape(1,-1),
                # y.reshape(1,-1), Y_norm_squared=None, squared=False, X_norm_squared=None))
        clust.append(((np.dot(x,x) + np.dot(y,y) - 2* np.dot(x,y))**0.5)*1.0)
    k =np.argmin(clust)
    centriole[k] = centriole[k] + y/float(len(clusters[k]))
    clusters[k].append(datasetT.filenames[i])
print(len(clusters))
for i in range(true_k):
        for j in clusters[i]:
                print(j)    
        print()



#outfile = "out.txt"
#np.savetxt(outfile,(np.hstack((dataset.filenames[125:],tmp))) ,delimiter = ',')   

