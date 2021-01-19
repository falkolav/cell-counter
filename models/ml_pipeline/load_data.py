import os
import numpy as np
from PIL import Image


def load_images(folder):
    images = np.array([np.array(Image.open(folder + i), dtype='float32') for i in os.listdir(folder)])
    labels = np.array([np.float32(i.split('.')[-2]) for i in os.listdir(folder)])
    return images, labels