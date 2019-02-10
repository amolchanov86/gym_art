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


a = 0.3 * np.array([1., 1.])
b = 0.3 * np.array([1., 1., 1.])
filt = SecondOrdFilter(a=a,b=b)
time_s = 1
freq = 100
x = np.ones([time_s * freq])

xf = np.zeros([time_s * freq])
steps = np.linspace(0,time_s, time_s * freq)
for t in range(time_s * freq-1):
    xf[t+1] = filt.step(x[t])

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
delay = 0.15
filt = FirstOrdFilter(T=delay, dt=1./freq)
for t in range(time_s * freq-1):
    xf[t+1] = filt.step(x[t])

plt.plot(steps, x, label="x")
plt.plot(steps, xf, label="x_filt")
plt.axvline(x=delay)
plt.legend()
plt.show(block=False)


x_sin = np.sin(2*np.pi*freq/8*steps)
x_sin_f = np.zeros_like(x_sin)
for t in range(time_s * freq-1):
    x_sin_f[t+1] = filt.step(x_sin[t])

plt.figure(2)
plt.plot(steps, x_sin, label="x")
plt.plot(steps, x_sin_f, label="x_filt")
plt.axvline(x=delay)
plt.legend()
plt.show(block=False)

# num = [1]
# den = [0.075 1]
# signal.TransferFunction(num, den, dt=0.005)

# f_samp = 200
# time = 1 #s
# t_num = f_samp * time
# t = np.linspace(0,1,t_num)
# x = copy(t)
# x[int(t_num/2):] = 1.


input("Enter ...")