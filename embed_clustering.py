import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from facenet_my import model
from keras.utils import Progbar
from tensorflow.keras.applications import resnet
from facenet_my import train_data
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage
import config


cluster_x = []
prog_bar = Progbar(target=50)

for n, (anchor, pos, neg) in enumerate(train_data.take(50)):
    batch_anchor_embedding = model.embedding(resnet.preprocess_input(anchor))
    batch_pos_embedding = model.embedding(resnet.preprocess_input(pos))
    batch_neg_embedding = model.embedding(resnet.preprocess_input(neg))
    cluster_x.append(batch_anchor_embedding.numpy())
    cluster_x.append(batch_neg_embedding.numpy())
    cluster_x.append(batch_neg_embedding.numpy())
    prog_bar.update(n)


cluster_x = np.array(cluster_x).reshape((-1, config.EMBED_DIM))
print("\n", cluster_x.shape)

pca = PCA(n_components=50).fit_transform(cluster_x)
tsne = TSNE(n_components=2).fit_transform(pca)

ax = plt.subplots()
plt.scatter(tsne[:, 0], tsne[:, 1])
plt.xlabel("features")
plt.ylabel("features")
plt.title("Visualization of the clusters")
plt.show()

kmeans = KMeans().fit_transform(tsne)

print(kmeans.shape)
