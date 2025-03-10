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


K = 4
def h(x):
    return x

def f(x):
    if x > 0:
        return 8*x/100
    return 0

def O(x):
    if x > 0:
        return K + 2*x
    return 0

N_STATES = 6
N_ACTIONS = N_STATES

actions = {}

for s in range(N_STATES):
    actions[s] = []
    for a in range(N_STATES - s):
        actions[s].append(a)

delta = 0.01

R = {}
C = {}
P = {}

demand = [0.3,0.2,0.2,0.2,0.05,0.05]



for s in range(N_STATES):
    l = len(actions[s])
    R[s] = np.zeros(l)
    C[s] = np.zeros(l)
    P[s] = {}
    for a in actions[s]:
        C[s][a] = O(a) + h(s+a)
        P[s][a] = np.zeros(N_STATES)
        for d in range(N_STATES):
            s_ = s + a - d
            if s_ < 0:
                s_ = 0
            elif s_ > N_STATES - 1:
                s_ = N_STATES - 1
                
            P[s][a][s_] += demand[d]
        R[s][a] = 0
        # for s_ in range(N_STATES):
        #     R[s][a] += P[s][a][s_]*f(s_)
        
for s in range(N_STATES):
    for a in actions[s]:        
      for d in range(N_STATES):
            s_ = min(max(0,s+a-d),N_STATES-1)
            if s + a - d >= 0:
                R[s][a] += P[s][a][s_]*f(d)
            else:
                R[s][a] += 0
            


r_max = R[0][0]
c_max = C[0][0]


for s in range(N_STATES):
    for a in actions[s]:
        if C[s][a] > c_max:
            c_max = C[s][a]
        if R[s][a] > r_max:
            r_max = R[s][a]

for s in range(N_STATES):
    for a in actions[s]:
        C[s][a] = C[s][a]/c_max
        R[s][a] = R[s][a]/r_max

EPISODE_LENGTH = 7
CONSTRAINT = EPISODE_LENGTH/2

Ph = {}
Rh = {}
Ch = {}
for h in range(EPISODE_LENGTH):
    Ph[h] = P
    Rh[h] = R
    Ch[h] = C


C_b = CONSTRAINT/5  #Change this if you want different baseline policy.

NUMBER_EPISODES = 4e5
NUMBER_SIMULATIONS = 5


util_methods_1 = utils(delta, Ph,Rh,Ch,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,CONSTRAINT,C_b)
opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con = util_methods_1.compute_opt_LP_Constrained(0)
opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon = util_methods_1.compute_opt_LP_Unconstrained(0)
f = open('solution-in-h.pckl', 'wb')
pickle.dump([opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon], f)
f.close()


util_methods_1 = utils(delta, Ph,Rh,Ch,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,C_b,C_b)
policy_b, value_b, cost_b, q_b = util_methods_1.compute_opt_LP_Constrained(0)
f = open('base-in-h.pckl', 'wb')
pickle.dump([policy_b, value_b, cost_b, q_b], f)
f.close()




f = open('model-in-h.pckl', 'wb')
pickle.dump([NUMBER_SIMULATIONS, NUMBER_EPISODES, Ph, Rh, Ch, CONSTRAINT, N_STATES, N_ACTIONS, actions, EPISODE_LENGTH, delta], f)
f.close()

print('*******')
