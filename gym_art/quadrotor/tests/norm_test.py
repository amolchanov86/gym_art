#!/usr/bin/env python
import numpy as np
import time


v = np.array([1,2,3])

st = time.time()
np.linalg.norm(v)
np.linalg.norm(v)
np.linalg.norm(v)
np.linalg.norm(v)
np.linalg.norm(v)
print("time: ", time.time() - st)


st = time.time()
(v[0]**2 + v[1]**2 + v[2]**2)**0.5
(v[0]**2 + v[1]**2 + v[2]**2)**0.5
(v[0]**2 + v[1]**2 + v[2]**2)**0.5
(v[0]**2 + v[1]**2 + v[2]**2)**0.5
(v[0]**2 + v[1]**2 + v[2]**2)**0.5
print("time: ", time.time() - st)