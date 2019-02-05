#!/usr/bin/env python
import numpy as np
import time

def cross(v1,v2):
    return np.array([[0, -v1[2], v1[1]], [v1[2], 0, -v1[0]], [-v1[1], v1[0], 0]]) @ v2

def cross4(V1,V2):
    x1 = cross(V1[0,:],V2[0,:])
    x2 = cross(V1[1,:],V2[1,:])
    x3 = cross(V1[2,:],V2[2,:])
    x4 = cross(V1[3,:],V2[3,:])
    return np.array([x1,x2,x3,x4])

def cross_vec_mx4(V1,V2):
    x1 = cross(V1,V2[0,:])
    x2 = cross(V1,V2[1,:])
    x3 = cross(V1,V2[2,:])
    x4 = cross(V1,V2[3,:])
    return np.array([x1,x2,x3,x4])

bla1 = np.random.normal(size=[4,3])
bla2 = np.random.normal(size=[4,3])
vec = np.random.normal(size=[3,])

print("Time comparison of MX4 x MX4 cross products:")
st = time.time()
r1 = np.cross(bla1,bla2)
r1 = np.cross(bla1,bla2)
r1 = np.cross(bla1,bla2)
r1 = np.cross(bla1,bla2)
r1 = np.cross(bla1,bla2)
print(time.time()-st)
st = time.time()
r2 = cross4(bla1,bla2)
r2 = cross4(bla1,bla2)
r2 = cross4(bla1,bla2)
r2 = cross4(bla1,bla2)
r2 = cross4(bla1,bla2)
print(time.time()-st)
print("r1\n",r1,"\n")
print("r2\n",r2)


print("Time comparison of VEC x MX4 cross products:")
st = time.time()
r1 = np.cross(vec,bla2)
r1 = np.cross(vec,bla2)
r1 = np.cross(vec,bla2)
r1 = np.cross(vec,bla2)
r1 = np.cross(vec,bla2)
print(time.time()-st)
st = time.time()
r2 = cross_vec_mx4(vec,bla2)
r2 = cross_vec_mx4(vec,bla2)
r2 = cross_vec_mx4(vec,bla2)
r2 = cross_vec_mx4(vec,bla2)
r2 = cross_vec_mx4(vec,bla2)
print(time.time()-st)
print("r1\n",r1,"\n")
print("r2\n",r2)