"""
My view:
step 1. We use actor_net(nenual network) to predict action's probability, and the input of nenual network is current state;
step 2. We use (step 1)'s output, action probability, to random choose a action,then do this action and get the reward and the next_state;
step 3. We use critic_net(nenual network) to predict state's value,so we can get current_state_value=critic_net(current_state) and next_state_value=critic_net(next_state)
step 4. Use (step 3)'s output to calculate td_error={(reward+gamma*next_state_value)-current_state_value};
step 5. We can use square(td_error) as the loss to train the critic_net, it is easy to understand that the td_error is the prediction error of critic_net, so our target is to decrease the td_error;
step 6. We can use the actor_net's output(action probability) and current state's real action to form the current cross entropy;
step 7. We use current cross entropy multiply current_state_value(predicted by critic_net) to form a sample's loss;
step 8. Finally, we can use (step 7)'s sample loss train our actor_net.

Attention:
1. Update actor_net and critic_new every step ;
2. The actor_net have the same function of nenual network in Policy Gradient;
3. We can use critic_net to predict the current_state_value which can used to form loss of actor_net, so we can say that critic will guide the actor_net how to export real action probability.
"""


import numpy as np
import gym

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

np.random.seed(2)

# Superparameters
MAX_EPISODE = 3000
# DISPLAY_REWARD_THRESHOLD = 200
DISPLAY_REWARD_THRESHOLD = 50  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
# env = gym.make('MountainCar-v0')
env.seed(1)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):
    def __init__(self, n_features, n_actions, lr=0.001):
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self._build_net()
        
    def _build_net(self):
        # ------------------ build actor_net -----------------
        num_hiddens = 20
        self.actor_net=nn.Sequential(
            nn.Linear(self.n_features, num_hiddens),
            nn.ReLU(inplace=True),
            nn.Linear(num_hiddens, self.n_actions), 
            nn.Softmax()
            )
        
        # weight init
        for name,params in self.actor_net.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.1)
            else:
                init.normal_(params, mean=0., std=0.1)
        
        # define optimizer
        self.optimizer =torch.optim.Adam(self.actor_net.parameters(),lr=self.lr)

    def learn(self, s, a, td):
        state = torch.tensor(s[np.newaxis, :], dtype=torch.float)
        action = torch.tensor(a, dtype=torch.long)
        td_error = torch.tensor(td, dtype=torch.float)
        
        # forward
        self.actor_net.train()
        acts_prob = self.actor_net(state) # every action's probability
        
        # loss
        log_prob = torch.log(acts_prob[0,action]) # - Cross Entropy
        exp_v = log_prob*td_error
        loss = -exp_v
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return exp_v

    def choose_action(self, s):
        state = torch.tensor(s[np.newaxis, :], dtype=torch.float)
        
        self.actor_net.eval()
        acts_prob = self.actor_net(state).detach().numpy()
        
        action = np.random.choice(np.arange(acts_prob.shape[1]), p=acts_prob.ravel())
        
        return action

class Critic(object):
    def __init__(self, n_features, lr=0.01):
        self.n_features = n_features
        self.lr = lr
        self._build_net()
        
    def _build_net(self):
        # ------------------ build actor_net -----------------
        num_hiddens = 20
        self.critic_net=nn.Sequential(
            nn.Linear(self.n_features, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, 1), 
            )
        
        # weight init
        for name,params in self.critic_net.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.1)
            else:
                init.normal_(params, mean=0., std=0.1)
        
        # define optimizer
        self.optimizer =torch.optim.Adam(self.critic_net.parameters(),lr=self.lr)

    def learn(self, s, r, s_):
        state = torch.tensor(s[np.newaxis, :], dtype=torch.float)       
        state_next = torch.tensor(s_[np.newaxis, :], dtype=torch.float)
        reward = torch.tensor(r, dtype=torch.float)
        
        self.critic_net.eval()
        value_next = self.critic_net(state_next)
        
        # forward
        self.critic_net.train()
        value = self.critic_net(state)
        
        # loss
        td_error = reward + GAMMA*value_next - value
        loss = td_error*td_error
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return td_error      

actor = Actor(n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: 
            env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: 
            r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_).detach().numpy()  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True  # rendering
            
            print("episode:", i_episode, "  reward:", int(running_reward))
            break

