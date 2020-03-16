import math
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import skimage.color as color

from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table
from skimage.transform import rotate

import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color
from skimage import measure


image1 = plt.imread('Capture.PNG')
image1 = color.rgb2gray(image1)
thresh = filters.threshold_otsu(image1, 10)
binary = image1 < thresh
plt.imshow(binary, cmap=plt.cm.gray)
plt.show()
label_img = label(binary)
regions = regionprops(label_img)
fig, ax = plt.subplots()
ax.imshow(binary, cmap=plt.cm.gray)


for props in regions:
    y0, x0 = props.centroid
   # ax.plot(x0, y0, '.g', markersize=10)
    print("x0: " + str(x0) + "   y0: " + str(y0))
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)
    #image1[int(y0),int(x0)] =1

#print (image1)
#plt.imshow(image1)
plt.show()

