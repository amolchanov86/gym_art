#!/usr/bin/env python

from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

num = [6.0705967e-8]
den = [1. -0.9745210]

f = 333.3
dt = 1./f
t = np.linspace(0,1,int(f))
x = np.sin(2*np.pi*100*t)
xf = signal.lfilter(num, den, x, axis=0)




##################################
from scipy.optimize import minimize
import copy

class SecondOrdFilter(object):
    """
    Filter:
    x -> y
    v[t] = x[t] - a[0]v[t-1] - a[1]*v[t-2]
    y[t] = b[0]v[t] - b[1]v[t-1] - b[2]*v[t-2]
    """
    def __init__(self, a, b):
        self.a = copy.copy(a)
        self.b = copy.copy(b)
        self.v = [0., 0., 0.]
    
    def step(self,x):
        # import pdb;pdb.set_trace()
        v1_next = self.v[0]
        v2_next = self.v[1]
        self.v[0] = x - self.a[0]*self.v[1] - self.a[1]*self.v[2]
        y = self.b[0]*self.v[0] + self.b[1]*self.v[1] + self.b[2]*self.v[2]
        self.v[1],self.v[2] = v1_next,v2_next
        return y


class SecondOrdFilter(object):
    """
    Filter:
    x -> y
    v[t] = x[t] - a[0]v[t-1] - a[1]*v[t-2]
    y[t] = b[0]v[t] - b[1]v[t-1] - b[2]*v[t-2]
    """
    def __init__(self, a, b):
        self.a = copy.copy(a)
        self.b = copy.copy(b)
        self.v = [0., 0., 0.]
    
    def step(self,x):
        # import pdb;pdb.set_trace()
        v1_next = self.v[0]
        v2_next = self.v[1]
        self.v[0] = x - self.a[0]*self.v[1] - self.a[1]*self.v[2]
        y = self.b[0]*self.v[0] + self.b[1]*self.v[1] + self.b[2]*self.v[2]
        self.v[1],self.v[2] = v1_next,v2_next
        return y

class FirstOrdFilter(object):
    """
    Filter:
    x -> y
    y[t] = dt/T*x + y[t-1] - dt/T
    """
    def __init__(self, T, dt):
        self.T = T
        self.dt = dt
        self.y_prev = 0.
        self.tau = np.clip(4 * self.dt/self.T, a_min=0., a_max=1.)
    
    def step(self, x):
        y =  self.tau * (x - self.y_prev) + self.y_prev
        self.y_prev = y
        return y

class FirstOrdHistFilter(object):
    """
    Filter:
    x -> y
    y[t] = dt/T*x + y[t-1] - dt/T
    """
    def __init__(self, Tup, Tdown, dt):
        self.Tup = Tup
        self.Tdown = Tdown
        self.dt = dt
        self.y_prev = 0.
        self.tau_up = np.clip(4 * self.dt/self.Tup, a_min=0., a_max=1.)
        self.tau_down = np.clip(4 * self.dt/self.Tdown, a_min=0., a_max=1.)
    
    def step(self, x):
        dx = x - self.y_prev
        if dx > 0:
            y =  self.tau_up * (x - self.y_prev) + self.y_prev
        else:
            y =  self.tau_down * (x - self.y_prev) + self.y_prev
        self.y_prev = y
        return y

class GazeboFilter(object):
    def __init__(self, Tup=0.0125, Tdown=0.025, dt=0.01):
        self.Tup = Tup
        self.Tdown = Tdown
        self.x_prev = 0
        self.alpha_up = np.exp(-dt/self.Tup)
        self.alpha_down = np.exp(-dt/self.Tdown)

    def step(self, x):
        if x > self.x_prev:
            y = self.alpha_up*self.x_prev + (1-self.alpha_up)*x
        else:
            y = self.alpha_down*self.x_prev + (1-self.alpha_down)*x
        self.x_prev = y
        return y



def filter_fitness(coeff):
    delay = 0.15
    max_val = 1.04
    a = coeff[:2]
    b = coeff[2:]

    filt = SecondOrdFilter(a=a,b=b)
    time_s = 2
    freq = 200
    x = np.ones([time_s * freq])
    xf = np.zeros([time_s * freq])
    steps = np.linspace(0,time_s, time_s * freq)
    for t in range(time_s * freq-1):
        xf[t+1] = filt.step(x[t])
    
    delay_step = int(delay * freq)
    t_peak = np.argmax(xf)
    
    xf_slope = xf[:delay_step-1] - 1
    # xf_after_slope = np.sum(1 - xf[delay_step:])
    # xf_after_slope_abs = 1./(10) * np.sum(np.abs(1 - xf[-10:]))
    xf_slope[xf_slope < -1]  = -xf_slope[xf_slope < -1]
    slope_int = 1./delay_step * np.sum(xf_slope)
    # print(slope_int, delay_step, xf[:delay_step])

    return 0.1 * np.abs(t_peak - delay_step) + 0.1*np.abs(1. - xf[delay_step]) + np.abs(max_val - xf[t_peak])+ 0.01*np.sum(1. - xf[delay_step:delay_step+20]) + 0.5*np.abs(1. - xf[-1]) + slope_int


#######################################################
delay_up = 0.15
delay_down = 2
delay_pause = 2.6
time_s = 4.5
freq = 500
x = np.ones([int(time_s * freq)])
x[int((delay_pause)*freq):] = 0.
xf = np.zeros([int(time_s * freq)])
steps = np.linspace(0,time_s, int(time_s * freq))


a = 0.3 * np.array([1., 1.])
b = 0.3 * np.array([1., 1., 1.])
filt = SecondOrdFilter(a=a,b=b)

for t in range(int(time_s * freq-1)):
    xf[t+1] = filt.step(x[t])
thrust = xf**2

coeff_init = 0.3 * np.ones([5])
coeff_init[1] = 0.0001
coeff_init[-1] = 0.001


# coeff_init = np.array([[-8.02566556e-01,  2.09232652e-04,  1.58155885e-02,  1.87442987e-01,  2.87674206e-04]])
# coeff_init = np.array([[-3.99674190e-01  3.04578896e-04  6.75242981e-10  6.03194337e-01 -2.01605501e-03]])

## Best:
# coeff_init = np.array([-4.78091860e-01,  5.03318489e-03,  1.27318499e-10,  5.33259382e-01, -6.30305732e-03])

# res = minimize(filter_fitness, coeff_init, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
# print(res.x)
# filt = SecondOrdFilter(a=res.x[:2],b=res.x[2:])

# coeff_opt = res.x
# coeff_opt = np.array([-4.78091860e-01,  5.03318489e-03,  1.27318499e-10,  5.33259382e-01, -6.30305732e-03])


# filt = SecondOrdFilter(a=coeff_opt[:2],b=coeff_opt[2:])

# filt = FirstOrdFilter(T=delay, dt=1./freq)
import time
filt = FirstOrdHistFilter(Tup=delay_up, Tdown=delay_down, dt=1./freq)

st = time.time()
for t in range(int(time_s * freq-1)):
    xf[t+1] = filt.step(x[t])
thrust = xf**2
print("Hist filter time: ", time.time() - st)

delay_up_thrust = 0.2
delay_down_thrust = 1.5
filt_thrust = FirstOrdHistFilter(Tup=delay_up_thrust, Tdown=delay_down_thrust, dt=1./freq)

delay_up_gaz = 0.0325 #0.0125
delay_down_gaz = 0.5 #0.025
filt_gaz = GazeboFilter(Tup=delay_up_gaz, Tdown=delay_down_gaz, dt=1./freq)
thrust_filt_approx = np.zeros_like(thrust)
xf_gaz = np.zeros_like(thrust)

# thrust_filt_approx[t+1] = filt_thrust.step(x[t])

st = time.time()
for t in range(int(time_s * freq-1)):
    xf_gaz[t+1] = filt_gaz.step(x[t])
thrust_gaz = xf_gaz**2
print("Gaz filter time: ", time.time() - st)

plt.plot(steps, x, label="vel")
plt.plot(steps, xf, label="vel_filt")
plt.plot(steps, thrust, label="thrust")
# plt.plot(steps, thrust_filt_approx, label="thrust_approx")
plt.plot(steps, thrust_gaz, label="thrust_gazebo")
plt.axvline(x=delay_up, color="red")
plt.axvline(x=delay_down + delay_pause, color="red")



x_sin = np.sin(2*np.pi*freq/8*steps)
x_sin_f = np.zeros_like(x_sin)
for t in range(int(time_s * freq-1)):
    x_sin_f[t+1] = filt.step(x_sin[t])

# plt.figure(2)
# plt.plot(steps, x_sin, label="x")
# plt.plot(steps, x_sin_f, label="x_filt")
# plt.axvline(x=delay_up, color="red")
# plt.legend()
# plt.show(block=False)

# num = [1]
# den = [0.075 1]
# signal.TransferFunction(num, den, dt=0.005)

# f_samp = 200
# time = 1 #s
# t_num = f_samp * time
# t = np.linspace(0,1,t_num)
# x = copy(t)
# x[int(t_num/2):] = 1.


delay_up = 0.0325
delay_down = 0.5
t_switch = int((delay_pause)*freq)
steps_num = int(time_s * freq-1)
rot_thrust = np.zeros([int(time_s * freq),4])
rot_thrust[t_switch:,:2] = 0.
rot_thrust[:t_switch,:2] = 1.
rot_thrust[t_switch:,2:] = 1.
rot_thrust[:t_switch,2:] = 0.

rot_vel = rot_thrust**0.5
rot_vel_filt = np.zeros_like(rot_vel)
rot_vel_filt[:10,2:] = 1.

def filter_step(rot, rot_prev, alpha_up, alpha_down):
    alpha = alpha_up * np.ones([4,])
    alpha[rot < rot_prev] = alpha_down
    return alpha * rot_prev + (1.-alpha)*rot

for t in range(steps_num):
    rot_vel_filt[t+1] = filter_step(rot_vel[t], rot_vel_filt[t], 
        alpha_up=np.exp(-1./freq / delay_up), 
        alpha_down=np.exp(-1/ freq / delay_down))
rot_thrust_filt = rot_vel_filt**2
    
plt.figure(3)
for i in range(4):
    plt.plot(steps, rot_thrust_filt[:,i], label="thrust_%d" % i)

plt.legend()
plt.show(block=False)






input("Enter ...")