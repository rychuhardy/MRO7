import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from pyclustering.cluster.kmedians import kmedians

# https://pypi.python.org/pypi/pyclustering
# https://github.com/annoviko/pyclustering/blob/master/pyclustering/cluster/examples/kmedians_examples.py
img = io.imread('AlaskaLynx_EN-US9313111559_1920x1080.jpg', as_grey=False)
dd = img.reshape(1920*1080, 3)

# optionally remove duplicates

kmedians_instance = kmedians(dd, [[255, 0, 0], [255, 255,0], [0,0,255], [0,255,255]])
kmedians_instance.process()
clusters = kmedians_instance.get_clusters()
print(img.shape)