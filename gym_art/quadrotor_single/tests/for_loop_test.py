#!/usr/bin/env python
import numpy as np
import time

def step1():
    mx = np.random.rand(3,3)
    v = np.random.rand(3)
    return mx @ v


def step10():
    step1()
    step1()
    step1()
    step1()

def step10_forloop():
    [step1() for i in range(4)]


iter = 1000

ts = time.time()
for i in range(iter):
    step10()
time_manual = time.time() -  ts

ts = time.time()
for i in range(iter):
    step10_forloop()

time_forloop = time.time() - ts

print("Manual: ", time_manual, " Forloop:", time_forloop, "Ratio: ", time_forloop/time_manual)
