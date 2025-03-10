import numpy as np
import pulp as p
import time
import math
import sys
from tqdm import tqdm

class utils:
    def __init__(self,delta, P,R,C,EPISODE_LENGTH,N_STATES,N_ACTIONS,ACTIONS,CONSTRAINT,Cb):
        self.P = P.copy()
        self.R = R.copy()
        self.C = C.copy()
        self.EPISODE_LENGTH = EPISODE_LENGTH
        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.ACTIONS = ACTIONS
        self.delta = delta
        self.delta_p = self.delta/(16*(self.N_STATES**2)*self.N_ACTIONS*EPISODE_LENGTH)
        
        self.P_hat = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.P_sampled = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES))
        
        self.NUMBER_OF_OCCURANCES = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS))
        self.NUMBER_OF_OCCURANCES_p = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES))

        self.alpha = 0.1 * np.ones([self.N_ACTIONS, self.N_STATES, self.N_STATES])
        self.lmbda = 0.0

        self.beta_prob = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES))
        self.beta_prob_1 = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS))
        self.beta_prob_2 = {} #np.zeros((self.EPS_LENGTH,self.N_STATES,self.N_ACTIONS))
        self.beta_prob_T = {}
        self.Psparse = [[[] for i in self.ACTIONS] for j in range(self.N_STATES)]
        
        self.mu = np.zeros(self.N_STATES)
        self.mu[0] = 1.0
        self.CONSTRAINT = CONSTRAINT
        self.Cb = Cb

        for s in range(self.N_STATES):
            for a in self.ACTIONS[s]:
                for s_1 in range(self.N_STATES):
                    if self.P[0][s][a][s_1] > 0:
                            self.Psparse[s][a].append(s_1)

        for h in range(self.EPISODE_LENGTH):
            
            self.P_hat[h] = {}
            self.NUMBER_OF_OCCURANCES[h] = {}
            self.beta_prob_1[h] = {}
            self.beta_prob_2[h] = {}
            self.beta_prob_T[h] = {}
            self.NUMBER_OF_OCCURANCES_p[h] = {}
            self.beta_prob[h] = {}
            
            for s in range(self.N_STATES):
                self.P_hat[h][s] = {}
                l = len(self.ACTIONS[s])
                
                self.NUMBER_OF_OCCURANCES[h][s] = np.zeros(l)
                self.beta_prob_1[h][s] = np.zeros(l)
                self.beta_prob_2[h][s] = np.zeros(l)
                self.beta_prob_T[h][s] = np.zeros(l)
                self.NUMBER_OF_OCCURANCES_p[h][s] = np.zeros((l, N_STATES))
                self.beta_prob[h][s] = np.zeros((l, N_STATES))
            
                for a in self.ACTIONS[s]:
                    self.P_hat[h][s][a] = np.zeros(self.N_STATES)
                    
    def step(self,s, a, h):
        probs = np.zeros((self.N_STATES))
        for next_s in range(self.N_STATES):
            probs[next_s] = self.P[h][s][a][next_s]
        next_state,obj_c,con_c = int(np.random.choice(np.arange(self.N_STATES),1,replace=True,p=probs)),self.R[h][s][a],self.C[h][s][a]
        self.alpha[a,s, next_state] += 1
        return next_state,obj_c,con_c

    def setCounts(self,ep_count_p,ep_count):
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    self.NUMBER_OF_OCCURANCES[h][s][a] += ep_count[h,s, a]
                    for s_ in range(self.N_STATES):
                        self.NUMBER_OF_OCCURANCES_p[h][s][a, s_] += ep_count_p[h,s, a, s_]

    def compute_confidence_intervals(self,ep, mode):
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    if self.NUMBER_OF_OCCURANCES[h][s][a] == 0:
                        self.beta_prob[h][s][a, :] = np.ones(self.N_STATES)
                        self.beta_prob_T[h][s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[h][s][a],1))
                    else:
                        if mode == 2:
                            self.beta_prob[h][s][a, :] = min(np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[h][s][a], 1)), 1)*np.ones(self.N_STATES)
                        elif mode == 3:
                            self.beta_prob_T[h][s][a] = np.sqrt(ep/max(self.NUMBER_OF_OCCURANCES[h][s][a],1))
                        for s_1 in range(self.N_STATES):
                            if mode == 0:
                                self.beta_prob[h][s][a,s_1] = min(np.sqrt(ep*self.P_hat[h][s][a][s_1]*(1-self.P_hat[h][s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[h][s][a],1)) + ep/(max(self.NUMBER_OF_OCCURANCES[h][s][a],1)), ep/(max(np.sqrt(self.NUMBER_OF_OCCURANCES[h][s][a]),1)), 1)
                            elif mode == 1:
                                self.beta_prob[h][s][a, s_1] = min(2*np.sqrt(ep*self.P_hat[h][s][a][s_1]*(1-self.P_hat[h][s][a][s_1])/max(self.NUMBER_OF_OCCURANCES[h][s][a],1)) + 14*ep/(3*max(self.NUMBER_OF_OCCURANCES[h][s][a],1)), 1)
                    self.beta_prob_1[h][s][a] = max(self.beta_prob[h][s][a, :])
                    self.beta_prob_2[h][s][a] = sum(self.beta_prob[h][s][a, :])
            
    def update_empirical_model(self,ep):
        P = {}
        for s in range(self.N_STATES):
            P[s] = {}
            for a in self.ACTIONS[s]:
                P[s][a] = np.zeros(self.N_STATES)
                out = np.random.dirichlet(self.alpha[a, s])
                for sn in range(self.N_STATES):
                    P[s][a][sn] = out[sn]

        for h in range(self.EPISODE_LENGTH):
            self.P_sampled[h] = P
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    if self.NUMBER_OF_OCCURANCES[h][s][a] == 0:
                        self.P_hat[h][s][a] = 1/self.N_STATES*np.ones(self.N_STATES)
                    else:
                        for s_1 in range(self.N_STATES):
                            self.P_hat[h][s][a][s_1] = self.NUMBER_OF_OCCURANCES_p[h][s][a,s_1]/self.NUMBER_OF_OCCURANCES[h][s][a]
                        self.P_hat[h][s][a] /= np.sum(self.P_hat[h][s][a])
                    if abs(sum(self.P_hat[h][s][a]) - 1)  >  0.001:
                        print("empirical is wrong")

    def compute_lang_opt(self,k):
        nu_C = (self.CONSTRAINT-self.Cb)*self.EPISODE_LENGTH
        nu = nu_C*np.sqrt(k)

        eps_C = .05 * (self.EPISODE_LENGTH**1.5) * np.sqrt(((self.N_STATES)**2)*self.N_ACTIONS)
        delta = k*self.N_STATES*self.N_ACTIONS*self.EPISODE_LENGTH

        eps = (eps_C*(1 + np.log(delta)))/ (np.sqrt(k * np.log(delta)))

        R = {}
        for h in range(self.EPISODE_LENGTH):
            R[h] = {}
            for s in range(self.N_STATES):
                R[h][s] = {}
                for a in self.ACTIONS[s]:
                    R[h][s][a] = self.R[h][s][a] + (self.lmbda/nu)*self.C[h][s][a]

        policy,cost_of_policy = self.valueiteration(self.P_sampled,R)

        self.lmbda = max(0,self.lmbda + cost_of_policy[0][0] + eps - self.CONSTRAINT)
        return policy

    def compute_lang_opt1(self,k,NUMBER_EPISODES):
        nu_C = (self.CONSTRAINT-self.Cb)*self.EPISODE_LENGTH
        nu = nu_C*np.sqrt(k)

        eps_C = 20 * (self.EPISODE_LENGTH**2) * np.sqrt(((self.N_STATES)**3)*self.N_ACTIONS)
        
        eps = (eps_C*(1 + np.log(k/self.delta_p)))/ (np.sqrt(k * np.log(k/self.delta_p)))

        L = math.log(NUMBER_EPISODES/self.delta_p)

        newC = {}
        beta = {}
        totR = {}

        for h in range(self.EPISODE_LENGTH):
            newC[h] = {}
            beta[h] = {}
            totR[h] = {}

            for s in range(self.N_STATES):
                newC[h][s] = {}
                beta[h][s] = {}
                totR[h][s] = {}
                for a in self.ACTIONS[s]:
                    beta[h][s][a] = np.sqrt(L/max(self.NUMBER_OF_OCCURANCES[h][s][a],1 ))
                    newC[h][s][a] = self.C[h][s][a] - (1 + self.EPISODE_LENGTH*self.N_STATES)*beta[h][s][a]
                    totR[h][s][a] = self.R[h][s][a] - (1 + self.EPISODE_LENGTH*self.N_STATES)*beta[h][s][a] + (self.lmbda/nu)*newC[h][s][a]
    
        policy,newcost_of_policy = self.valueiteration1(self.P_hat,totR,newC)
        
        self.lmbda = max(0,self.lmbda + max(newcost_of_policy[0][0],0) + eps - self.CONSTRAINT)
        return policy

    def valueiteration(self,P,R):
        qVals = {} 
        qMax = {}
        opt_policy = np.zeros((self.EPISODE_LENGTH,self.N_STATES)) 

        qMax[self.EPISODE_LENGTH] = np.zeros(self.N_STATES, dtype=np.float32)
        for i in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - i - 1
            qMax[j] = np.zeros(self.N_STATES, dtype=np.float32)
    
            for s in range(self.N_STATES):
                qVals[j, s] = np.zeros(len(self.ACTIONS[s]), dtype=np.float32)
                for a in self.ACTIONS[s]:
                    qVals[j,s][a] = R[j][s][a] + np.dot(P[j][s][a], qMax[j + 1])

                opt_policy[j][s] = int(np.argmin(qVals[j,s]))
                qMax[j][s] = np.min(qVals[j,s])

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(P, self.deterministic2stationary(opt_policy), self.R, self.C)
        return opt_policy,cost_of_policy

    def valueiteration1(self,P,R,C):
        qVals = {} 
        qMax = {}
        opt_policy = np.zeros((self.EPISODE_LENGTH,self.N_STATES)) 

        qMax[self.EPISODE_LENGTH] = np.zeros(self.N_STATES, dtype=np.float32)
        for i in range(self.EPISODE_LENGTH):
            j = self.EPISODE_LENGTH - i - 1
            qMax[j] = np.zeros(self.N_STATES, dtype=np.float32)
    
            for s in range(self.N_STATES):
                qVals[j, s] = np.zeros(len(self.ACTIONS[s]), dtype=np.float32)
                for a in self.ACTIONS[s]:
                    qVals[j,s][a] = R[j][s][a] + np.dot(P[j][s][a], qMax[j + 1])

                opt_policy[j][s] = int(np.argmin(qVals[j,s]))
                qMax[j][s] = np.min(qVals[j,s])

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(P, self.deterministic2stationary(opt_policy), self.R, C)
        return opt_policy,cost_of_policy

    def compute_opt_LP_Unconstrained(self, ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMinimize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS)) #[h,s,a]
    
        #create problem variables
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')
        
        #Objective function
        list_1 = [self.R[h][s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
        list_2 = [q[(h,s,a)] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]

        opt_prob += p.lpDot(list_1,list_2)
                
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[h-1][s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0
        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001, msg = 0))
                    
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                    probs = opt_policy[s,h,:]
                                                                  
        if ep != 0:
            return opt_policy, 0, 0, 0
                                                                          
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                         
        return opt_policy, value_of_policy, cost_of_policy, q_policy
                                                                                                                                                   
    def compute_opt_LP_Constrained(self, ep):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMinimize)
        opt_q = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS)) #[h,s,a]
                                                                                  
        q_keys = [(h,s,a) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]
                                                                                          
        q = p.LpVariable.dicts("q",q_keys,lowBound=0,cat='Continuous')

        opt_prob += p.lpSum([q[(h,s,a)]*self.R[h][s][a] for  h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]])
            
        opt_prob += p.lpSum([q[(h,s,a)]*self.C[h][s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s]]) - self.CONSTRAINT <= 0
            
        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                q_list = [q[(h,s,a)] for a in self.ACTIONS[s]]
                pq_list = [self.P[h-1][s_1][a_1][s]*q[(h-1,s_1,a_1)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1]]
                opt_prob += p.lpSum(q_list) - p.lpSum(pq_list) == 0

        for s in range(self.N_STATES):
            q_list = [q[(0,s,a)] for a in self.ACTIONS[s]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0

        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.001, msg = 0))
                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    opt_q[h,s,a] = q[(h,s,a)].varValue
                for a in self.ACTIONS[s]:
                    if np.sum(opt_q[h,s,:]) == 0:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                    else:
                        opt_policy[s,h,a] = opt_q[h,s,a]/np.sum(opt_q[h,s,:])
                        if math.isnan(opt_policy[s,h,a]):
                            opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                        elif opt_policy[s,h,a] > 1.0:
                            print("invalid value printing")
                                                                                                                                               
        if ep != 0:
            return opt_policy, 0, 0, 0
        
        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                          
        return opt_policy, value_of_policy, cost_of_policy, q_policy

    def compute_extended_LP(self,ep, cb):
        opt_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_ACTIONS)) #[s,h,a]
        opt_prob = p.LpProblem("OPT_LP_problem",p.LpMinimize)
        opt_z = np.zeros((self.EPISODE_LENGTH,self.N_STATES,self.N_ACTIONS,self.N_STATES)) #[h,s,a,s_]
        #create problem variables
        
        z_keys = [(h,s,a,s_1) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
        z = p.LpVariable.dicts("z_var",z_keys,lowBound=0,upBound=1,cat='Continuous')
            
        r_k = {}
        for h in range(self.EPISODE_LENGTH):
            r_k[h] = {}
            for s in range(self.N_STATES):
                l = len(self.ACTIONS[s])
                r_k[h][s] = np.zeros(l)
                for a in self.ACTIONS[s]:
                    r_k[h][s][a] = self.R[h][s][a] - self.EPISODE_LENGTH**2/(self.CONSTRAINT - cb)* self.beta_prob_2[h][s][a]

        opt_prob += p.lpSum([z[(h,s,a,s_1)]*r_k[h][s][a] for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]])
        #Constraints
        opt_prob += p.lpSum([z[(h,s,a,s_1)]*(self.C[h][s][a] + self.EPISODE_LENGTH*self.beta_prob_2[h][s][a]) for h in range(self.EPISODE_LENGTH) for s in range(self.N_STATES) for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]) - self.CONSTRAINT <= 0

        for h in range(1,self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                z_list = [z[(h,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
                z_1_list = [z[(h-1,s_1,a_1,s)] for s_1 in range(self.N_STATES) for a_1 in self.ACTIONS[s_1] if s in self.Psparse[s_1][a_1]]
                opt_prob += p.lpSum(z_list) - p.lpSum(z_1_list) == 0
                                                                                                                                                                                                              
        for s in range(self.N_STATES):
            q_list = [z[(0,s,a,s_1)] for a in self.ACTIONS[s] for s_1 in self.Psparse[s][a]]
            opt_prob += p.lpSum(q_list) - self.mu[s] == 0
                                                                                                                                                                                                                                                                                                                                                                                                                             #start_time = time.time()
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_prob += z[(h,s,a,s_1)] - (self.P_hat[h][s][a][s_1] + self.beta_prob[h][s][a,s_1]) *  p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                        opt_prob += -z[(h,s,a,s_1)] + (self.P_hat[h][s][a][s_1] - self.beta_prob[h][s][a,s_1])* p.lpSum([z[(h,s,a,y)] for y in self.Psparse[s][a]]) <= 0
                                                                                                                                                                                                                                        
        status = opt_prob.solve(p.PULP_CBC_CMD(fracGap=0.01, msg = 0))
                                                                                                                                                                                                                                      
        if p.LpStatus[status] != 'Optimal':
            return np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), np.zeros((self.N_STATES, self.EPISODE_LENGTH)), p.LpStatus[status], np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS))
                                                                                                                                                                                                                                                  
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                for a in self.ACTIONS[s]:
                    for s_1 in self.Psparse[s][a]:
                        opt_z[h,s,a,s_1] = z[(h,s,a,s_1)].varValue
                        if opt_z[h,s,a,s_1] < 0 and opt_z[h,s,a,s_1] > -0.001:
                            opt_z[h,s,a,s_1] = 0
                        elif opt_z[h,s,a,s_1] < -0.001:
                            print("invalid value")
                            sys.exit()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        den = np.sum(opt_z,axis=(2,3))
        num = np.sum(opt_z,axis=3)
                                                                                                                                                                                                                                                                                                                                                                                                                                     
        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                sum_prob = 0
                for a in self.ACTIONS[s]:
                    opt_policy[s,h,a] = num[h,s,a]/den[h,s]
                    sum_prob += opt_policy[s,h,a]
                if abs(sum(num[h,s,:]) - den[h,s]) > 0.0001:
                    print("wrong values")
                    #print(sum(num[h,s,:]),den[h,s])
                    sys.exit()
                if math.isnan(sum_prob):
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = 1/len(self.ACTIONS[s])
                else:
                    for a in self.ACTIONS[s]:
                        opt_policy[s,h,a] = opt_policy[s,h,a]/sum_prob

        q_policy, value_of_policy, cost_of_policy = self.FiniteHorizon_Policy_evaluation(self.P, opt_policy, self.R, self.C)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        return opt_policy, value_of_policy, cost_of_policy, p.LpStatus[status], q_policy

                                                                                                                                                                                                                                      
    def FiniteHorizon_Policy_evaluation(self,Px,policy,R,C):
        
        q = np.zeros((self.N_STATES,self.EPISODE_LENGTH, self.N_ACTIONS))
        v = np.zeros((self.N_STATES, self.EPISODE_LENGTH))
        c = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        P_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH,self.N_STATES))
        R_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))
        C_policy = np.zeros((self.N_STATES,self.EPISODE_LENGTH))

        for s in range(self.N_STATES):
            x = 0
            for a in self.ACTIONS[s]:
                x += policy[s, self.EPISODE_LENGTH - 1, a]*C[self.EPISODE_LENGTH - 1][s][a]
            c[s,self.EPISODE_LENGTH-1] = x #np.dot(policy[s,self.EPISODE_LENGTH-1,:], self.C[s])

            for a in self.ACTIONS[s]:
                q[s, self.EPISODE_LENGTH-1, a] = R[self.EPISODE_LENGTH - 1][s][a]
            v[s,self.EPISODE_LENGTH-1] = np.dot(q[s, self.EPISODE_LENGTH-1, :], policy[s, self.EPISODE_LENGTH-1, :])

        for h in range(self.EPISODE_LENGTH):
            for s in range(self.N_STATES):
                x = 0
                y = 0
                for a in self.ACTIONS[s]:
                    x += policy[s,h,a]*R[h][s][a]
                    y += policy[s,h,a]*C[h][s][a]
                R_policy[s,h] = x
                C_policy[s,h] = y
                for s_1 in range(self.N_STATES):
                    z = 0
                    for a in self.ACTIONS[s]:
                        z += policy[s,h,a]*Px[h][s][a][s_1]
                    P_policy[s,h,s_1] = z #np.dot(policy[s,h,:],Px[s,:,s_1])

        for h in range(self.EPISODE_LENGTH-2,-1,-1):
            for s in range(self.N_STATES):
                c[s,h] = C_policy[s,h] + np.dot(P_policy[s,h,:],c[:,h+1])
                for a in self.ACTIONS[s]:
                    z = 0
                    for s_ in range(self.N_STATES):
                        z += Px[h][s][a][s_] * v[s_, h+1]
                    q[s, h, a] = R[h][s][a] + z
                v[s,h] = np.dot(q[s, h, :],policy[s, h, :])

        return q, v, c


    def  deterministic2stationary(self, policy):
        stationary_policy = np.zeros((self.N_STATES, self.EPISODE_LENGTH, self.N_ACTIONS))
        for s in range(self.N_STATES):
            for h in range(self.EPISODE_LENGTH):
                a = int(policy[h, s])
                stationary_policy[s, h, a] = 1

        return stationary_policy