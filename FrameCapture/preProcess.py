import time
from ScreenViewer import ScreenViewer
import matplotlib.pyplot as plt
import numpy as np
from skimage import data
from skimage import io
from skimage.exposure import histogram
import skimage.data as data
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage.filters import try_all_threshold


if __name__ == "__main__":
    img = plt.imread('Capture.JPG')
    # image = rgb2gray(img)
    image = color.rgb2gray(img)
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    plt.imshow(binary, cmap=plt.cm.gray)
    plt.show()



