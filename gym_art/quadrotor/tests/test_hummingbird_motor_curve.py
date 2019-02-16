#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# See https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds

## Comparing normalized forces and revolutions vs actual ones
a = 4.9782e-08
b = 7.7151e-07
w = np.linspace(0,6000,100)
plt.plot(w, a*w**2 + a*w)
a_ = 0.0476
w_ = np.linspace(0.,1.,100)
plt.plot(w_ * 6000, 1.75 * ((1-a_)*w_**2 + a_*w_))