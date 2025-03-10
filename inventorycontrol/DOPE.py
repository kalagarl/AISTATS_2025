import numpy as np
import pandas as pd
from utils import utils
import matplotlib.pyplot as plt
import time
import os
import math
import pickle
import sys
import random
from tqdm import tqdm


start_time = time.time()
RUN_NUMBER = 4

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#Initialize:
f = open('model-in.pckl', 'rb')
[NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, N_ACTIONS,actions, EPISODE_LENGTH, DELTA] = pickle.load(f)
f.close()

f = open('solution-in.pckl', 'rb')
[opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon] = pickle.load(f)
f.close()

f = open('base-in.pckl', 'rb')
[pi_b, val_b, cost_b, q_b] = pickle.load(f)
f.close()

Cb = cost_b[0, 0]
print(CONSTRAINT - Cb)

K0 = N_STATES**2*N_ACTIONS*EPISODE_LENGTH**3/((CONSTRAINT - Cb)**2)

NUMBER_EPISODES = int(NUMBER_EPISODES)
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
ConRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
WConRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
WObjRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))

NUMBER_INFEASIBILITIES = np.zeros((NUMBER_SIMULATIONS, NUMBER_EPISODES))

L = math.log(6 * N_STATES**2 * EPISODE_LENGTH * NUMBER_EPISODES / DELTA)#math.log(2 * N_STATES * EPISODE_LENGTH * NUMBER_EPISODES * N_STATES**2 / DELTA)

for sim in range(NUMBER_SIMULATIONS):
    print(sim)
    print('------------------------------------------')
    util_methods = utils(DELTA,P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,CONSTRAINT,Cb)
    ep_count = np.zeros((N_STATES, N_STATES))
    ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES))
    objs = []
    cons = []
    wcons = []
    wobjs = []
    for episode in tqdm(range(NUMBER_EPISODES)):
        
        if episode <= K0:
            pi_k = pi_b
            val_k = val_b
            cost_k = cost_b
            q_k = q_b
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0)
            util_methods.compute_confidence_intervals(L, 1)
        else:
            util_methods.setCounts(ep_count_p, ep_count)
            util_methods.update_empirical_model(0)
            util_methods.compute_confidence_intervals(L, 1)
            pi_k, val_k, cost_k, log, q_k = util_methods.compute_extended_LP(0, Cb)
            if log != 'Optimal':  #Added this part to resolve issues about infeasibility. Because I am not sure about the value of K0, this condition would take care of that
                pi_k = pi_b
                val_k = val_b
                cost_k = cost_b
                q_k = q_b
                print(log)

        if episode == 0:
            ObjRegret2[sim, episode] = abs(val_k[0, 0] - opt_value_LP_con[0, 0])
            ConRegret2[sim, episode] = max(0, cost_k[0, 0] - CONSTRAINT)
            WConRegret2[sim, episode] = cost_k[0, 0] - CONSTRAINT
            WObjRegret2[sim, episode] = opt_value_LP_con[0, 0] - val_k[0, 0]

            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            wcons.append(WConRegret2[sim, episode])
            wobjs.append(WObjRegret2[sim, episode])

            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = 1
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + abs(val_k[0, 0] - opt_value_LP_con[0, 0])
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + max(0, cost_k[0, 0] - CONSTRAINT)
            WConRegret2[sim, episode] = WConRegret2[sim, episode - 1] + cost_k[0, 0] - CONSTRAINT
            WObjRegret2[sim, episode] = WObjRegret2[sim, episode - 1] + opt_value_LP_con[0, 0] - val_k[0, 0]

            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
            wcons.append(WConRegret2[sim, episode])
            wobjs.append(WObjRegret2[sim, episode])

            if cost_k[0, 0] > CONSTRAINT:
                NUMBER_INFEASIBILITIES[sim, episode] = NUMBER_INFEASIBILITIES[sim, episode - 1] + 1
        
        ep_count = np.zeros((N_STATES, N_STATES))
        ep_count_p = np.zeros((N_STATES, N_STATES, N_STATES))

        s = 0
        for h in range(EPISODE_LENGTH):
            prob = pi_k[s, h, :]
            a = int(np.random.choice(util_methods.N_ACTIONS, 1, replace = True, p = prob))
            next_state,obj_c,con_c = util_methods.step(s, a, h)
            
            ep_count[s, a] += 1
            ep_count_p[s, a, next_state] += 1
            s = next_state


    f = open('dope_woriginal_regret_4e5_5_run4.pckl', 'wb')
    pickle.dump([ObjRegret2,ConRegret2,WObjRegret2,WConRegret2], f)
    f.close()

        
ObjRegret_mean = np.mean(ObjRegret2, axis = 0)
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)

WObjRegret_mean = np.mean(WObjRegret2, axis = 0)
WConRegret_mean = np.mean(WConRegret2, axis = 0)
WObjRegret_std = np.std(WObjRegret2, axis = 0)
WConRegret_std = np.std(WConRegret2, axis = 0)

title = 'DOPE' + str(RUN_NUMBER)
plt.figure()
plt.plot(range(NUMBER_EPISODES), ObjRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ObjRegret_mean - ObjRegret_std, ObjRegret_mean + ObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Objective Regret')
plt.title(title)
plt.show()

plt.figure()
plt.plot(range(NUMBER_EPISODES), ConRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), ConRegret_mean - ConRegret_std, ConRegret_mean + ConRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('Constraint Regret')
plt.title(title)
plt.show()

plt.figure()
plt.plot(range(NUMBER_EPISODES), WObjRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), WObjRegret_mean - WObjRegret_std, WObjRegret_mean + WObjRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('WObjective Regret')
plt.title(title)
plt.show()

plt.figure()
plt.plot(range(NUMBER_EPISODES), WConRegret_mean)
plt.fill_between(range(NUMBER_EPISODES), WConRegret_mean - WConRegret_std, WConRegret_mean + WConRegret_std, alpha = 0.5)
plt.grid()
plt.xlabel('Episodes')
plt.ylabel('WConstraint Regret')
plt.title(title)
plt.show()