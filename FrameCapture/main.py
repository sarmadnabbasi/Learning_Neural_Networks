import time
from ScreenViewer import ScreenViewer
import matplotlib.pyplot as plt
from skimage.util import crop
import numpy as np
import skimage.filters as filters
import skimage.draw as draw
import skimage.color as color

if __name__ == "__main__":
    x = np.zeros((10,))
    y = np.zeros((10,))
    print(x)
    print(y)
    sv = ScreenViewer()
    sv.GetHWND('Unity 2018.4.14f1 Personal - PowerfulMagnet (modify).unity - UnityMagnets-master - PC, Mac & Linux Standalone <DX11>')
    sv.Start()
    while True:
        I, fps = sv.GetScreen()
        print(fps)
        I = crop(I, ((230, 430), (750, 758), (0, 0)), copy=False)
        I = color.rgb2gray(I)
        thresh = filters.threshold_otsu(I)
        binary = I > thresh
        plt.figure(1)
        plt.clf()
        plt.imshow(I, cmap=plt.cm.binary)
        plt.pause(0.01)

    # time.sleep(2)
    # sv.Stop()
