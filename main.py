import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from pyclustering.cluster.kmedians import kmedians
from sklearn.decomposition import PCA

# https://pypi.python.org/pypi/pyclustering
# https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/examples/kmedians_examples.py
img = np.array(io.imread('photo.jpg', as_grey=False), dtype="int64")
img_reshaped = img.reshape(180*180, 3)

# optionally remove duplicates
k = 20
np.random.shuffle(img_reshaped)
start_points = img_reshaped[:k]

kmedians_instance = kmedians(img_reshaped, start_points)
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
medians = kmedians_instance.get_medians()
print(img.shape)


pca = PCA(n_components=2)
reduced = pca.fit_transform(img_reshaped)

kmedians_result = sum([list(zip(idx, np.tile(c, (len(idx), 1)))) for idx, c in list(zip(clusters, medians))], [])
kmedians_X = np.array([reduced[idx] for idx, c in kmedians_result])
kmedians_colors = [c/255 for idx, c in kmedians_result]

plt.clf()
fig, ax = plt.subplots(figsize=(16,16))
ax.scatter(kmedians_X[:, 0], kmedians_X[:, 1], marker='s', s=50, color=kmedians_colors)

plt.show()