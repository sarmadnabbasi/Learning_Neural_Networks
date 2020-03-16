import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
from skimage import measure



# Construct some test data
x, y = np.ogrid[-np.pi:np.pi:100j, -np.pi:np.pi:100j]
r = np.sin(np.exp((np.sin(x)**3 + np.cos(y)**2)))

# Find contours at a constant value of 0.8
#contours = measure.find_contours(r, 0.8)

image = plt.imread('Capture.PNG')
image_g = color.rgb2gray(image)
contours = measure.find_contours(image_g, 0.8)
plt.imshow(image)
plt.show()
# Display the image and plot all contours found
fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)


print("Total Contours: " +str(n))
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()