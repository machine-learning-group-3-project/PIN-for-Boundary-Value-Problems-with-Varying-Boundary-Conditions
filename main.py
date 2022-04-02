# Machine Learning Project
# Spring 2022
# Authors: Melissa Butler,
# PINN to solve differential equations

# For the ODE: F(x, u(x), u'(x), u''(x)) = 0  on (a,b)
# With boundary conditions (BC): u(a) = u(b) = 0
# Find approximate solution: unet(x) = phi(x)N(x)
# where phi(x) = (b-x)(x-a) to capture BC
# and N(x) is the "transfer function of a suitably tuned neural network"

import numpy as np
from numpy import random
import math
from scipy.optimize import minimize


# Set up ODE and BC
F = lambda x, u, up, upp : upp-9*u
a = 0
b = 1

# Discretize Domain: n equally space training points 
n = 4
x = np.linspace(a,b,n)

# Set up neural network (1 hidden layer) 
q = 3                                   # Units in hidden Layer
t = random.rand(q)                      #translation/bias vector
w = random.rand(q)                      #1st weights
v = random.rand(q)                      #2nd weights
sigma = lambda z : z                    # TODO - not sure about activation function
#sigma = lambda z : 1/(1-math.exp(-z))   

# Find Residual for all x_i in xx
# R(x) = F(x, unet, unet', unet'') 
#      = F(x, phi*N, phi'*N+phi*N', phi''*N+phi'*N'+phi'*N+phi*N'') By Product Rule
N = np.zeros(n)
for i in range(n):                  # TODO - can we vectorize this?
    N[i] = v@sigma(x[i]*w+t)   
Np = v@w                            # TODO - double check derivatives (they seem too simple)
Npp = 0
phi = (b-x)*(x-a)
phi_p =  b - 2*x + a
phi_pp = -2
unet = phi*N
unetp = phi_p*N + phi*Np
unetpp = phi_pp*N + phi_p*Np + phi_p*N + phi*Npp

R = F(x, unet, unetp, unetpp)

# Total Loss (minimize this)
E = np.sum(R**2)                    #TODO - find t,w,v that minimizes E ---- For gradient descent we might need N_u and N_v

