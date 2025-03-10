
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils_h import utils
import sys
import pickle
import time
import pulp as p
import math
from copy import copy

c_0 = 1
c_1 = 1

mu_1 = 0.9
mu_2 = 0.1

BUFFER_SIZE = 1.0
lambda_m = 0.8

def P_1(a):
    if a == 0:
        return mu_1*lambda_m + (1-mu_1) *(1-lambda_m)
    else:
        return mu_2*lambda_m + (1-mu_2) * (1-lambda_m)

def P_2(a):
    if a == 0:
        return mu_1*(1-lambda_m)
    else:
        return mu_2*(1-lambda_m)

def P_3(a):
    if a == 0:
        return (1-mu_1)*lambda_m 
    else:
        return (1-mu_2)*lambda_m
    
def objcost(x):
    if x == 0:
        return c_0
    return 0

def concost(a):
    if a == 0:
        return c_1
    return 0

delta = 0.01
buffer_values = np.arange(0,BUFFER_SIZE,0.1)

N_STATES = len(buffer_values)
N_ACTIONS = 2

P = {}
R = {}
C = {}
actions = {}
for s in buffer_values:
   l = int(np.round(s/0.1))
   P[l] = {}
   R[l] = {}
   C[l] = {}
   actions[l] = [0,1]
   
   for a in range(0,2):
    P[l][a] = np.zeros(len(buffer_values))
    for s_1 in buffer_values:
        m = int(np.round(s_1/0.1))
        P[l][a][m] = 0
    R[l][a] = objcost(s)
    C[l][a] = concost(a)

    next_s = s
    P[l][a][l] += P_1(a)
    
    next_s = s+0.1
    if(next_s > BUFFER_SIZE-0.1):
        next_s = BUFFER_SIZE - 0.1
    m = int(np.round(next_s/0.1))
    P[l][a][m] += P_2(a)
    
    next_s = s-0.1
    if(next_s < 0):
        next_s = 0
    m = int(np.round(next_s/0.1))
    P[l][a][m] += P_3(a)

EPISODE_LENGTH = 10
CONSTRAINT = EPISODE_LENGTH/2

C_b = CONSTRAINT/5   #Change this if you want a different baseline policy.

NUMBER_EPISODES = 4e5
NUMBER_SIMULATIONS = 5

Ph = {}
Rh = {}
Ch = {}
for h in range(EPISODE_LENGTH):
    Ph[h] = P
    Rh[h] = R
    Ch[h] = C

util_methods_1 = utils(delta, Ph,Rh,Ch,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,CONSTRAINT,C_b)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0)
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0)
f = open('solution-queue-h.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()

util_methods_1 = utils(delta, Ph,Rh,Ch,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,C_b,C_b)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base-queue-h.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()

f = open('model-queue-h.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, Ph, Rh, Ch, CONSTRAINT, N_STATES, N_ACTIONS, actions, EPISODE_LENGTH, delta], f)
f.close()

print('*******')