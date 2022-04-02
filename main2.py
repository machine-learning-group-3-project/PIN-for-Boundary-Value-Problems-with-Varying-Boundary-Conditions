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
import matplotlib.pyplot as plt

def loss_function(weights):
    # Set up ODE and BC
    F = lambda x, u, up, upp : upp-9*u
    a = 0
    b = 1

    # Discretize Domain: n equally space training points 
    n = 4
    x = np.linspace(a,b,n)
    # Set up neural network (1 hidden layer) 
    q = 3                                   # Units in hidden Layer
    t = weights[:q]                         #translation/bias vector
    w = weights[q:2*q]                      #1st weights
    v = weights[2*q:]                       #2nd weights
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
    return np.sum(R**2)                   

w0 = random.rand(3*3)
res = minimize(loss_function, w0, method='nelder-mead')
print(res.x) #res.x is the optimize weights [t w v]


#Plot results
n=4
q=3
xx = np.linspace(0,1,n)
utrue = (9/2)*(xx-1)*xx
N = np.zeros(n)
t = res.x[:q]
w = res.x[q:2*q]
v = res.x[2*q:]
for i in range(4):                  # TODO - can we vectorize this?
    N[i] = v@(xx[i]*w+t)   
uapprx = (1-xx)*(xx-0)*N

plt.plot(xx, utrue)
plt.plot(xx, uapprx)
