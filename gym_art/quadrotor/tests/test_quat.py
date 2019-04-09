#!/usr/bin/env python

import numpy as np
import time
import gym_art.quadrotor.quad_utils as qu
import transforms3d as t3d

iterations  = 1000

# Sampling rotation matrices
R_arr = []
for i in range(iterations):
    R_arr.append(qu.rand_uniform_rot3d())


# Testing t3d transform
t_start = time.time()
for i in range(iterations): 
    t3d.quaternions.mat2quat(R_arr[i])
print("T3D quat time: ", time.time() - t_start)


# Testing t3d transform
t_start = time.time()
for i in range(iterations): 
    qu.R2quat(R_arr[i])
print("QuadUtils quat time: ", time.time() - t_start)


# Matching test
for i in range(50):
    qt3d = t3d.quaternions.mat2quat(R_arr[i])
    qQU = qu.R2quat(R_arr[i])
    if not np.all(np.isclose(qt3d, qQU)):
        print("Non-matching T2d : QU", qt3d, qQU) 