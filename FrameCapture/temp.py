import numpy as np
from scipy import constants

u0 = constants.mu_0
pi = constants.pi
M = (np.power(0.001, 3))/u0
print("M: " + str(M))

mk = np.array([1, 0], dtype=np.float64)
mi = np.array([1, 0], dtype=np.float64)
mk = M*mk
mi = M*mi
rk = [0, 0]
ri = [0.00025, 0]


rkriSub = np.subtract(rk, ri)
rkriSubDet = np.linalg.norm(rkriSub)
nik = (np.subtract(rk, ri))/rkriSubDet


torque = (3*(np.cross(mk, nik)*np.dot(mi, nik)) - np.cross(mk, mi))/(np.power(rkriSubDet, 3))
torque = torque * u0/(4*pi)

# print(-15*nik*(np.dot(mk, nik)*np.dot(mi, nik)))

force = -15*nik*(np.dot(mk, nik)*np.dot(mi, nik)) + 3*nik*np.dot(mk, mi) + 3*(mk*np.dot(mi, nik) + mi*np.dot(mk, nik))
force = force * (1/np.power(rkriSubDet, 4)) * (u0/(4*pi))

print(force)