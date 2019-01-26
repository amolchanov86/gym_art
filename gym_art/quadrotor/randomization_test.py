#!/usr/bin/env python
from  gym_art.quadrotor import quadrotor as q
from gym_art.quadrotor import quad_utils as qu

noise_ratio = 0.01

cf_par = q.crazyflie_params()
noise = q.get_dyn_randomization_params(cf_par, noise_ratio=noise_ratio)
cf_pert_par = q.perturb_dyn_parameters(cf_par, noise, sampler="normal")

print("###################################################")
print("CrazyFlie params:")
qu.print_dic(cf_par)

print("###################################################")
print("CrazyFlie perturbed params with ratio %f:" % noise_ratio)
qu.print_dic(cf_pert_par)

print("###################################################")
print("Equivalence: ", cf_par == cf_pert_par)

