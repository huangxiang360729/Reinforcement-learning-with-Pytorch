"""
My view:
step 1. We use nenual network to predict action's probability, and the input of nenual network is current state;
step 2. We use (step 1)'s output(action probability) to random choose a action,then do this action and get the reward;
step 3. Store <state,action,reward> tuple to memory;
step 4. After a episode, we can get lots of <state,action,reward> tuple;
step 5. Because we have get the every action of this episode, so we can calculate every state's value;
step 6. We can use the nenual network's output(action probability) and current state's real action to form the current cross entropy, know that every state have a cross entropy)
step 7. We use current cross entropy multiply current state's value to form a sample's loss;
step 8. Finally, we can use all sample's loss train our nenual network. 

Attention:
1. Every episode, only update nenual network once;
2. We use the product of cross entropy and  state's value to form loss, this loss can guide the policy how to change;
3. We use the cross entropy to express the difference of predction(action's probability) and label(one_hot of real action).
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()
        
        self.sfm = nn.Softmax(dim=1)

        if output_graph:
            #from torch.utils.tensorboard import SummaryWriter
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter("logs/")
            
            dummy_input = torch.rand(1, self.n_features)
            tb_writer.add_graph(self.eval_net, dummy_input)

    def _build_net(self):
        # ------------------ build evaluate_net -----------------
        num_hiddens = 10
        self.net=nn.Sequential(
            nn.Linear(self.n_features, num_hiddens),
            nn.Tanh(),
            nn.Linear(num_hiddens, self.n_actions), 
            )
        
        # weight init
        for name,params in self.net.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.1)
            else:
                init.normal_(params, mean=0., std=0.3)
        
        # define optimizer
        self.optimizer =torch.optim.Adam(self.net.parameters(),lr=self.lr)
        
    def pg_one_hot(self, label, class_num):
        return torch.eye(class_num)[label,:] # covert 1D'indices to 2D'one_hot
    
    """
    we assume batchsize=1
    y: net's prediction for all action's probability, the net's input is state
    label: real action for state
    factor: the value of this state
    """
    def pg_loss(self, y, label, factor):
        prob_weights = self.sfm(y) # every action's probability
        ce_loss = - torch.log(prob_weights) * self.pg_one_hot(label,y.shape[1]) # CrossEntropyLoss, but not to reduce_mean
        factor = factor.unsqueeze(1)
        loss = torch.mean( y * factor ) # Policy Gradients

        return loss

    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # add batch dimension
        
        self.net.eval()
        output = self.net(torch.tensor(observation,dtype=torch.float)) # forward
        prob_weights = self.sfm(output).detach().numpy() #  probability of every action
        
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        
        return action
        
        
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # forward
        self.net.train()
        y = self.net(torch.tensor(np.vstack(self.ep_obs),dtype=torch.float))
        
        # action label 
        label = torch.tensor(np.array(self.ep_as),dtype=torch.long)
        
        # value factor ,$$value = state-action_t reward + gamma*state_{t+1} $$
        factor = torch.tensor(discounted_ep_rs_norm,dtype=torch.float)
        
        # loss
        loss = self.pg_loss(y,label,factor) 
        
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []    # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs



DISPLAY_REWARD_THRESHOLD = -2000  # renders environment if total episode reward is greater then this threshold

RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.995,
    # output_graph=True,
)

for i_episode in range(1000):

    observation = env.reset()

    while True:
        if RENDER: 
            env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases

        RL.store_transition(observation, action, reward)

        if done:
            # calculate running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: 
                RENDER = True     # rendering

            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()  # train

            if i_episode == 30:
                plt.plot(vt)  # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()

            break

        observation = observation_
