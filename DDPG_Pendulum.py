"""
My view:
step 1. We use actor_eval_net(nenual network) to predict current_action(not action's probability), and the input of actor_eval_net is current_state;
step 2. We use actor_eval_net choose a current_action(will add noise), then do this current_action and get the reward and the next_state;

step 3. Enqueue <current_state,current_action,reward,next_state> tuple to memory queue, if overflow, dequeue the earliest tuple;
setp 4. Every step, random take batchsize(e.g. 32) tuples from memory to train actor_eval_net and critic_eval_net;

step 5. train critic_eval_net:
            A. We use critic_eval_net(nenual network) to predict Q value, and the input of critic_eval_net is [state, action];
            B. We can critic_eval_net to calculate current_state_value=critic_eval_net([current_state, current_action]);
            C. But we use actor_target_net(like DQN's target_net) to get next_state's action(next_action), so next_action=actor_target_net(next_state);
            D. We use critic_target_net(like DQN's target_net) to get next_state's value(next_state_value), so next_state_value=critic_target_net([next_state, next_action]);
            E. We calculate td_error={(reward+gamma*next_state_value)-current_state_value};
            F. We can use square(td_error) as the all sample loss to train the critic_eval_net, it is easy to understand that the td_error is the prediction error of critic_eval_net, so our target is to decrease the td_error;
            G. We "soft" update critic_target_net: target_params={(1-TAU)*target_params + TAU*eval_params}, TAU is very small, like 0.01;

step 6. train actor_eval_net:
            A. We use actor_eval_net to predict current_state's action(current_action);
            B. We uer critic_eval_net to predict current_state's Q value(current_state_value)
            C. It is easy to understand that our actor_eval_net's target is to increase current_state_value, so we use "- current_state_value" to form actor_eval_net's loss;
            D. We use all sample loss train our actor_eval_net;
            E. We "soft" update actor_target_net: target_params={(1-TAU)*target_params + TAU*eval_params};

Attention:
1. Update actor_eval_net and critic_eval_net every step, we also soft update actor_target_net and critic_target_net every step;
2. DDPG combines Policy Gradient(predicts action directly, not predict action's probability) and Actor-Critic (use Critic to guide Actor's update);
3. When we train critic_eval_net, we use actor_target_net to get next_action and use critic_target_net to get next_state_value;
4. When we tarin actor_eval_net, we use critic_eval_net to get current_state_value;
"""

import numpy as np

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

import gym
import time


#####################  hyper parameters  ####################
MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
ENV_NAME = 'Pendulum-v0'


###############################  DDPG  ####################################
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound,dtype=torch.float)
        
        num_hiddens = 30
        
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, self.action_dim), 
            nn.Tanh()
            )
        
        # weight init
        for name,params in self.net.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.)
            else:
                init.normal_(params, mean=0., std=0.001)
        
    def forward(self, states): 
        actions = self.net(states)
        scaled_actions = actions * self.action_bound
        return scaled_actions
    
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        num_hiddens = 30
        
        self.params = nn.ParameterDict({
            'w1_s': nn.Parameter(torch.randn(self.state_dim, num_hiddens)*0.001),
            'w1_a': nn.Parameter(torch.randn(action_dim, num_hiddens)*0.001),
            'b1': nn.Parameter(torch.zeros(1,num_hiddens))
        })
        
        self.linear = nn.Linear(num_hiddens, 1)
        
        for name, params in self.linear.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.)
            else:
                init.normal_(params, mean=0., std=0.001)
        
    def forward(self, states, actions): 
        y1 = torch.mm(states, self.params['w1_s'])
        y2 = torch.mm(actions, self.params['w1_a'])
        y = torch.relu( y1 + y2 + self.params['b1'] )
        q = self.linear(y)
        return q
        

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        
        self._build_actor()
        self._build_critic()

    def _build_actor(self):
        # ------------------ build evaluate_net -----------------
        self.actor_eval_net = ActorNet(self.s_dim, self.a_dim, self.a_bound)
        
        # define optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor_eval_net.parameters(),lr=LR_A)

        # ------------------ build target_net ------------------
        self.actor_target_net = deepcopy(self.actor_eval_net)

    def _build_critic(self):
        # ------------------ build evaluate_net -----------------
        self.critic_eval_net = CriticNet(self.s_dim, self.a_dim)
        
        # define optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic_eval_net.parameters(),lr=LR_C)

        # ------------------ build target_net ------------------
        self.critic_target_net = deepcopy(self.critic_eval_net)
        
    def soft_replace(self):
        for t,e in zip(self.actor_target_net.parameters(),self.actor_eval_net.parameters()):
            # "t = (1 - TAU)* t + TAU * e" is error, very important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            t.data.copy_((1 - TAU)* t + TAU * e)
        for t,e in zip(self.critic_target_net.parameters(),self.critic_eval_net.parameters()):
            t.data.copy_((1 - TAU)* t + TAU * e)

    def choose_action(self, state):
        state = state[np.newaxis, :]    # single state
        tensor_state = torch.tensor(state,dtype=torch.float)
        tensor_action = self.actor_eval_net(tensor_state)
        
        return tensor_action.detach().numpy()[0] # [[a]] ===> [a]

    def learn(self):
        # soft target replacement
        self.soft_replace()
        
        # get batch from memory
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        
        tensor_batch_states = torch.tensor(bs,dtype=torch.float)
        tensor_batch_actions = torch.tensor(ba,dtype=torch.float)
        tensor_batch_rewards = torch.tensor(br,dtype=torch.float)
        tensor_batch_next_states = torch.tensor(bs_,dtype=torch.float)
        
        ######################update actor######################
        # actor forward
        tensor_predicted_batch_actions = self.actor_eval_net(tensor_batch_states)
        
        tensor_Q_critic_eval = self.critic_eval_net(tensor_batch_states, tensor_predicted_batch_actions)
        
        # actor loss
        actor_loss = - torch.mean(tensor_Q_critic_eval)
        
        # actor backward
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        ######################update critic######################
        # critic forward
        tensor_Q_eval = self.critic_eval_net(tensor_batch_states, tensor_batch_actions)

        tensor_batch_next_actions = self.actor_target_net(tensor_batch_next_states).detach()
        tensor_Q_next = self.critic_target_net(tensor_batch_next_states, tensor_batch_next_actions).detach()
        tensor_Q_target = tensor_batch_rewards + GAMMA * tensor_Q_next
        
        tensor_td_error = tensor_Q_target - tensor_Q_eval
        
        # critic loss
        critic_loss = torch.mean(tensor_td_error*tensor_td_error)
        
        # critic backward
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
          
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

        
        
###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.learn()
            
        if(i == MAX_EPISODES - 50):
            RENDER = True

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)
