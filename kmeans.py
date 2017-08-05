import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
centers = [[1, 1], [-1, -1], [1, -1]]
dataset = []
dataset_name = 'arrhythmia'
with open(dataset_name + '.data') as f:
		conteudo = f.readlines()
for x in conteudo:
	dataset.append(x.strip().split(','))
dataset = np.asarray(dataset).astype(np.float)

target = []
for x in dataset:
	target.append(x[4])
target = np.asarray(target).astype(np.int)
dataset =  np.delete(dataset, -1, axis=1)
estimators = {'k_means_2_clusters': KMeans(n_clusters=2),
			  'k_means_3_clusters': KMeans(n_clusters=3),
              'k_means_iris_16_clusters': KMeans(n_clusters=16)}


fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.suptitle(str(est.n_clusters) + " Clusters")
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    est.fit(dataset)
    labels = est.labels_

    ax.scatter(dataset[:, 3], dataset[:, 0], dataset[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Atributo 1')
    ax.set_ylabel('Atributo 2')
    ax.set_zlabel('Atributo 3')
    fignum = fignum + 1
plt.show()
