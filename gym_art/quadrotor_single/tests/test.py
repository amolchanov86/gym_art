import numpy as np
import time
#vel = np.array([100,99,88])
st = time.time()
vel_3 = ((0,0,0),(1,2,3),(4,5,6))
print(vel_3[1:3])
def func():
	for i in range(0,25):
		global vel_3
		vel = (1+i,2+i,3+i)
		vel_3 = vel_3[0:2]
		vel_3 = (vel,vel_3[0],vel_3[1])
func()		
print(time.time()-st)
print("------------")
print(vel_3)