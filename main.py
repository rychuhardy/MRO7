import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from pyclustering.cluster.kmedians import kmedians

# https://pypi.python.org/pypi/pyclustering
img = io.imread('AlaskaLynx_EN-US9313111559_1920x1080.jpg', as_grey=False)
dd = img.reshape(1920*1080, 3)

# optionally remove duplicates



print(img.shape)