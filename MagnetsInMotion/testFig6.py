import matplotlib.pyplot as plt
import numpy as np

y = np.zeros((7,))
x2=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
y2=[-38.0, -26.0, -18.3, -13.3, -10.0, -07.5, -05.8]
y3=[-49.5, -29.8, -19.2, -12.9, -09.1, -06.6, -04.9]
y3old=[-43.2, -21.9, -12.4, -07.7, -05.0, -03.5, -02.5]
y4=[-45.6, -04.5, -01.4, -00.6, -00.3, -00.2, -00.13]

fig = plt.figure()
plt.plot(x2, y2, label='1 Cell')
plt.plot(x2, y3old, label='2 Cell')
#plt.plot(x2, y3old)
#plt.plot(x2, y4)
plt.axis([0, 0.6, -50,0])
fig.suptitle('Total Force for Multi Cell Model', fontsize=10)
plt.xlabel('Distance (cm)', fontsize=10)
plt.ylabel('Force (N)', fontsize=10)
plt.legend();
plt.show()

