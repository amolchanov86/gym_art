import numpy as np
import numpy.linalg as norm
import time

omega_vec = [1.3,2.4,3.3]

st = time.time()

omega_norm = np.sqrt(np.cumsum(np.square(omega_vec)))[2]
print("Time comparison of cumsum",omega_norm)
print(time.time()-st)


st = time.time()
omega_norm1 = np.linalg.norm(omega_vec)
print("Time comparison of norm",omega_norm1)
print(time.time()-st)

v= omega_vec

st = time.time()
# (v[0]**2 + v[1]**2 + v[2]**2)**0.5
# (v[0]**2 + v[1]**2 + v[2]**2)**0.5
# (v[0]**2 + v[1]**2 + v[2]**2)**0.5
# (v[0]**2 + v[1]**2 + v[2]**2)**0.5
# (v[0]**2 + v[1]**2 + v[2]**2)**0.5
print("Check new",(v[0]**2 + v[1]**2 + v[2]**2)**0.5)

print("time: ", time.time() - st)