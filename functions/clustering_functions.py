import numpy as np
from umap import UMAP

from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
# import scanpy as sc
# import anndata
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import silhouette_samples,silhouette_score
# from sklearn.neighbors import kneighbors_graph

def calculate_dbscan(X,ind,eps=0.1,min_samples=50):
    
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    try:
      print("Silhouette Coefficient: %0.3f"
            % silhouette_score(X, labels))
    except:
      print('Silhouette impossible; only 1 cluster recognized')
    # add the index of each point in k to the corresoponding labels
    # labelsWithInd = np.zeros((len(labels),2)).astype(int)
    # labelsWithInd[ :,1] = labels
    # labelsWithInd[ :,0] = ind
    # return X,labels,core_samples_mask,labelsWithInd
    return X,labels,core_samples_mask

def calculate_umap(data,n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', rstate=42,dens=False):
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric, random_state=rstate,  densmap=dens,
        # verbose=True,
    )
    u = fit.fit_transform(data)
    
    return u


from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

def kmeans_fit(data,n_clusters=3,n_init=10,max_iter=300):
    kmeans = KMeans(
        init="random",
        random_state=42,
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        
    )
    data = StandardScaler().fit_transform(data)
    kmeans.fit(data)

    # # The lowest SSE value
    # print(kmeans.inertia_)

    # # Final locations of the centroid
    # print(kmeans.cluster_centers_)

    # # The number of iterations required to converge
    print(f'converged after {kmeans.n_iter_} iterations')

    return np.asarray(kmeans.labels_)
# labels = kmeans_fit(data = scaled_features,n_clusters=3,n_init=10,max_iter=300)

def highVal_ind(k,f,labels,max_ = True):
    arr = []
    for l in np.unique(labels):
        arr.append(k[ f].loc[[i for i,label in zip(k.index,labels) if label==l]].mean())
    wantedLabel = np.argmax(arr) if max_ else np.argmin(arr)
    # loaction in original index (k isnot reseted)
    ind = [i for i,label in zip(k.index,labels) if label==wantedLabel]
    # loaction in new index (according to labels of new index)
    ind2 = [i for i,j in enumerate(labels) if j==wantedLabel]
    return ind,ind2