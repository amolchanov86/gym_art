#!/usr/bin/env python
import numpy as np
import time

iterations = 1000

def K(kxyz):
    kx,ky,kz = kxyz[0],kxyz[1],kxyz[2]
    return np.array([
        [0., -kz, ky],
        [kz, 0., -kx],
        [-ky, kx, 0.]
    ])


def K2(kxyz):
    kx,ky,kz = kxyz[0],kxyz[1],kxyz[2]
    return np.array([
        [-kz**2-ky**2, kx*ky, kx*kz],
        [kx*ky, -kz**2-kx**2, kz*ky],
        [kx*kz, kz*ky, -ky**2 - kx**2]
    ])

kxyz = np.random.uniform(size=[iterations,3])

# Testing matrix multiplication
print("Evaluating K @ K ...")
t_start = time.time()
for i in range(iterations):
    Kval = K(kxyz[i,:])
    K2val = Kval @ Kval
t_end = time.time()
print("K @ K time for %d iter: %f" % (iterations, t_end - t_start))

print("Evaluating K2 ...")
# Testing matrix construction
t_start = time.time()
for i in range(iterations):
    K2val = K2(kxyz[i,:])

t_end = time.time()
print("K2 time for %d iter: %f" % (iterations, t_end - t_start))

print("Evaluating mismatch ...")
# Testing consistency
K2_mismatch = 0
for i in range(iterations):
    Kval = K(kxyz[i,:])
    K2val_mult = Kval @ Kval 

    K2val_direct = K2(kxyz[i,:])

    K2_mismatch += np.sum( np.abs(K2val_direct - K2val_mult) )

print("K2 avg mismatch %f" % (K2_mismatch / iterations))
