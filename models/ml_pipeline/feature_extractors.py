from skimage.feature import hog
from skimage.filters import frangi
import numpy as np


def extract_hog(X):
    hog_list = [hog(i, orientations=7, pixels_per_cell=(14, 14), cells_per_block=(1, 1), feature_vector=True, visualize=False) for i in X]
    X = np.reshape(hog_list, (len(X), -1))
    return X

def extract_frangi(X):
    X = np.array([frangi(i, beta=0.9, black_ridges=False) for i in X])
    return X