#!/usr/bin/env python
import numpy as np
import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import sys, os
import matplotlib.pyplot as plt

from gym_art.quadrotor.quad_utils import OUNoise

parser = argparse.ArgumentParser(
    description="Argument description: \n",
    formatter_class=ArgumentDefaultsHelpFormatter)
    # formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "-m","--mu",
    type=float, default=0.,
    help="Mean"
)
parser.add_argument(
    "-s","--sigma",
    type=float, default=0.01,
    help="Sigma - noise scale"
)
parser.add_argument(
    "-th","--theta",
    type=float, default=0.15,
    help="Theta - noise stabilization coefficient"
)
args = parser.parse_args()

ou = OUNoise(1, mu=args.mu, theta=args.theta, sigma=args.sigma)
states = []
for i in range(300):
    states.append(ou.noise())

plt.plot(states)
plt.show(block=False)
input("Press Enter to exit ...")