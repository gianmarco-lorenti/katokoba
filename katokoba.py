# Libs
from random import random, choices
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Global 
M = 20
N = 100
K = 5000
P = 30
tmax = int(5e4)

class KatoKoba:
    def __init__(self):
        self.tmax = tmax
        self.env = Environment()
        self.agent = Agent(self.tmax)
        # Histories
        self.s_hist = [] # states
        self.a_hist = [] # actions 
        self.r_hist = [] # rwds
        
    def run(self):
        next_state, next_action = None, None
        for t in range(self.tmax):
            if t == 0:
                current_state = self.env.initial_state()
                # Agent reads state, responds
                current_action = self.agent.policy(current_state)
            else:
                current_state = next_state
                current_action = next_action
            self.s_hist.append(current_state)
            self.a_hist.append(current_action)
            # Agent reads state, responds
            current_reward, next_state = self.env.update(current_action)
            self.r_hist.append(current_reward)
            next_action = self.agent.policy(next_state)
            # Perform Update
            SARSA = (current_state, current_action, current_reward, next_state, next_action)
            self.agent.update(SARSA)
            # if t % 100 == 0:
                # print(t)
            
            

class Environment:
    def __init__(self):
        self.N = N # state space size
        self.M = M # action space size
        self.P = P # aize of antigen pattern set (state db)
        self.q = .9 # get sick probability
        self.R0 = M/2 # reward threshold
        self.state_db, self.best_act_db = self._gen_db() # states and best actions
        # keys are strings, values are np.array
        self.st_ix = None # current state index
        self.st = None # current state
        self.act = None # action at current state
        self.rwd = None # current reward
   
    def initial_state(self):
        ran_ix = int(random()*self.P)
        self.st_ix = ran_ix
        self.st = np.ravel(self.state_db[ran_ix])
        return np.ravel(self.state_db[ran_ix])
   
    def update(self, action):
        """Update state, return reward and new state"""
        # compute reward
        self.rwd = self._reward(self.st_ix, action)
        # generate transition probs
        probs = [self._transition_probs(ixs, action) for ixs in range(len(self.state_db))]
        # select new state
        new_state_ix = choices(range(P), probs, k=1)[0]
        self.st_ix = new_state_ix
        self.st = np.ravel(self.state_db[new_state_ix])
        return self.rwd, self.st
    
    def _gen_db(self):
        count = 0
        db = {}
        while not len(db) == P:
            phato = np.zeros((N))
            probs = np.random.rand(len(phato))
            phato[probs < 0.5] = 1
            db[str(phato)] = phato
            count += 1
        db = np.array(list(db.values()))
        
        best_acts = np.zeros((P, M))
        probs = np.random.rand(*best_acts.shape)
        best_acts[probs < 0.5] = 1
      
        return db, best_acts

        
    def _reward(self, state_ix, action):
        """Reward as defined in paper"""
        return self.M - self._hamming(np.ravel(self.best_act_db[state_ix]), action) 
    
    def _hamming(self, v_1, v_2):
        """Hamming distance"""
        return np.sum(np.abs(v_1 - v_2))
    
    def _transition_probs(self, target_ix, action):
        """Transition probability of state with index target_ix"""
        if self.st_ix == 0: # if healthy
            if target_ix == 0:
                return 1 - self.q
            else:
                return self.q/(self.P-1)
        elif self.st_ix == target_ix:
            prob = (self.M - self._reward(self.st_ix, action))/(self.M - self.R0)
            return min(1, prob)
        elif target_ix == 0:
            prob = -(self.R0 - self._reward(self.st_ix, action))/(self.M - self.R0)
            return max(0,prob)
        else:
            return 0.

class Agent:
    """Represent Immune Network as the learning and decision-making entity"""
    def __init__(self, tmax):
        self.K = K
        self.M = M
        self.N = N
        self.alpha = 1e-1
        self.gamma = 0.75
        self.beta_min = 1
        self.beta_max = 20
        self.beta = self.beta_min
        self.activities = None
        self.n = np.ones((K,)) # parameter vector
        self.weights = np.random.normal(0, np.sqrt(2/N), (K, N)) # w_kj
        self.intens = np.random.normal(0, np.sqrt(2/K), (M, K)) # u_j
        self.sigmoid = lambda x: 1/(1+np.exp(-x))
        self.tmax = tmax
        # SARSA Variables
        self.current_state = None
        self.current_action = None
        self.rwd = None
        self.next_state = None
        self.next_action = None
        self.t = 0
        
    def update(self, sarsa):
        # read state transition
        self.current_state, self.current_action, self.rwd, self.next_state, self.next_action = sarsa
        # perform gradient descent step
        self._gd_step()
        self.t += 1
        self._beta_update()
    
    def _beta_update(self):
        """Linear update of expl. param"""
        self.beta = self.t*(self.beta_max - self.beta_min)/self.tmax + self.beta_min
        # print(self.beta)
    
    def _gd_step(self):
        """Perform GD step on running SARSA variables"""
        grad_1 = (self.rwd + self.gamma*self._q_approx(self.next_state, self.next_action) -\
                self._q_approx(self.current_state, self.current_action))
        features = self._compute_activities(self.current_state)*((self.intens.T).dot(self.current_action))
        #print("Grad1: ", grad_1) # returns float
# =============================================================================
#         print("Features: ", type(features)) # returns np.float64
#         print()
# =============================================================================
        gradient = features * grad_1 # vector gradient
        self.n *= (1 + self.alpha*gradient)
        # repair
        self.n[self.n < 0] = 0
        
    def policy(self, state):
        action = np.zeros((M,))
        probs = self.sigmoid(self.beta*self.intens.dot(self._compute_activities(state) * self.n))
        thresholds = np.random.rand(M)
        action[thresholds < probs] = 1
        return action
    
    def _compute_activities(self, state):
        # Return cytokine for each cell type k
        return  self.sigmoid(self.weights.dot(state.T))
    
    def _q_approx(self, state, action):
        """Return an estimate of Q-function parametrized by n"""
        features = self._compute_activities(state)*((self.intens.T).dot(action))
        #print(features.shape)
        q_approx = self.n.dot(features)
        #print("Features", type(features))
        return q_approx
    
    def _activity(self, k, state):
        """Compute and return cytokine activity of type-k cells on state"""
        stimulus = self.weights[k].dot(state)
        return self.sigmoid(stimulus)
    
iterrs = 1
r_hists = np.zeros((iterrs, tmax))
if __name__ == '__main__':
    for iterr in range(iterrs):
        model = KatoKoba()
        model.run()
        r_hist = model.r_hist[:]
        n = model.agent.n
        r_hists[iterr, :] = r_hist
        print(iterr)
        
# data = pd.DataFrame(r_hists)
# data.to_csv("r_hists_{}.csv".format(iterrs), sep=';')
rwds = np.mean(r_hists, axis=0)
plt.plot(rwds)