import numpy as np
import pandas as pd
from utils_h import utils
import matplotlib.pyplot as plt
import random
import time
import os
import math
import pickle
import sys
from tqdm import tqdm

start_time = time.time()
RUN_NUMBER = 2

random.seed(RUN_NUMBER)
np.random.seed(RUN_NUMBER)

#Initialize:
f = open('model-in-h.pckl', 'rb')
[NUMBER_SIMULATIONS, NUMBER_EPISODES, P, R, C, CONSTRAINT, N_STATES, N_ACTIONS, actions, EPISODE_LENGTH, DELTA] = pickle.load(f)
f.close()

f = open('solution-in-h.pckl', 'rb')
[opt_policy_con, opt_value_LP_con, opt_cost_LP_con, opt_q_con, opt_policy_uncon, opt_value_LP_uncon, opt_cost_LP_uncon, opt_q_uncon] = pickle.load(f)
f.close()

f = open('base-in-h.pckl', 'rb')
[pi_b, val_b, cost_b, q_b] = pickle.load(f)
f.close()

Cb = cost_b[0, 0]

NUMBER_EPISODES = int(NUMBER_EPISODES)
NUMBER_SIMULATIONS = int(NUMBER_SIMULATIONS)

ObjRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))
ConRegret2 = np.zeros((NUMBER_SIMULATIONS,NUMBER_EPISODES))

for sim in range(NUMBER_SIMULATIONS):
    print(sim)
    print('------------------------------------------')
    util_methods = utils(DELTA, P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,actions,CONSTRAINT,Cb)
    ep_count = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS))
    ep_count_p = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS, N_STATES))
    objs = []
    cons = []
    
    for episode in tqdm(range(NUMBER_EPISODES)):
        util_methods.setCounts(ep_count_p, ep_count)
        util_methods.update_empirical_model(0)
        
        pi_k = util_methods.compute_lang_opt1(episode+1,NUMBER_EPISODES)

        ep_count = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS))
        ep_count_p = np.zeros((EPISODE_LENGTH,N_STATES, N_ACTIONS, N_STATES))

        obj_cs,con_cs = 0,0

        s = 0
        for h in range(EPISODE_LENGTH):
            a = int(pi_k[h,s])
            #prob = pi_k[s, h, :]
            #a = int(np.random.choice(util_methods.N_ACTIONS, 1, replace = True, p = prob))
            next_state,obj_c,con_c = util_methods.step(s, a, h)
            
            obj_cs+=obj_c
            con_cs+=con_c
            ep_count[h,s, a] += 1
            ep_count_p[h,s, a, next_state] += 1
            
            s = next_state

        if episode == 0:
            ObjRegret2[sim, episode] =  opt_value_LP_con[0, 0] - obj_cs
            ConRegret2[sim, episode] = con_cs - CONSTRAINT
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])
        else:
            ObjRegret2[sim, episode] = ObjRegret2[sim, episode - 1] + opt_value_LP_con[0, 0] - obj_cs
            ConRegret2[sim, episode] = ConRegret2[sim, episode - 1] + con_cs - CONSTRAINT
            objs.append(ObjRegret2[sim, episode])
            cons.append(ConRegret2[sim, episode])

    f = open('oppd_regret-h-'+'4e5-'+'20-'+'05'+'.pckl', 'wb')
    pickle.dump([ObjRegret2,ConRegret2], f)
    f.close()
        
ObjRegret_mean = np.mean(ObjRegret2, axis = 0)
ConRegret_mean = np.mean(ConRegret2, axis = 0)
ObjRegret_std = np.std(ObjRegret2, axis = 0)
ConRegret_std = np.std(ConRegret2, axis = 0)



title = 'OPPD' + str(RUN_NUMBER)

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