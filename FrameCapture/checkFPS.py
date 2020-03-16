import time
from ScreenViewer import ScreenViewer
import matplotlib.pyplot as plt
from skimage.util import crop
import numpy as np
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

if __name__ == "__main__":
    y = np.zeros((30,))
    print(y)
    sv = ScreenViewer()
    sv.GetHWND('Magnets')
    sv.Start()
    for i in range(30):
        I, fps = sv.GetScreenWithTime()
        #print(i)
        y[i]=fps
        I = crop(I, ((230, 430), (750, 758), (0, 0)), copy=False)
        I = color.rgb2gray(I)
        thresh = filters.threshold_otsu(I)
        binary = I > thresh
        #plt.figure(1)
        #plt.clf()
        #plt.imshow(I, cmap=plt.cm.binary)
        #plt.pause(0.01)
        time.sleep(0.02)

    print(y[29]-y[0])
    y[0]=y[1]
    plt.figure(fps)
    plt.plot(y)
    plt.show()
    sv.Stop()
