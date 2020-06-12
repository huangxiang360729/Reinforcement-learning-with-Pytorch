import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

import gym

# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        # self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if output_graph:
            #from torch.utils.tensorboard import SummaryWriter
            from tensorboardX import SummaryWriter
            tb_writer = SummaryWriter("logs/")
            
            dummy_input = torch.rand(1, self.n_features)
            tb_writer.add_graph(self.eval_net, dummy_input)

        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net -----------------
        num_hiddens = 10
        self.eval_net=nn.Sequential(
            nn.Linear(self.n_features, num_hiddens),
            nn.ReLU(),
            nn.Linear(num_hiddens, self.n_actions), 
            )
        
        # weight init
        for name,params in self.eval_net.named_parameters():
            if 'bias' in name:
                init.constant_(params, val=0.1)
            else:
                init.normal_(params, mean=0., std=0.3)
        
        # define loss function
        self.loss= nn.MSELoss()
        
        # define optimizer
        self.optimizer =torch.optim.RMSprop(self.eval_net.parameters(),lr=self.lr)
        

        # ------------------ build target_net ------------------
        self.target_net = deepcopy(self.eval_net)
    
    def replace_target_op(self):
        self.target_net.parameters = deepcopy(self.eval_net.parameters)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension, add a new dimension
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            self.eval_net.eval()
            actions_value = self.eval_net(torch.tensor(observation,dtype=torch.float))

            action = np.argmax(actions_value.detach().numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_op()
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # use target_net get q_next (old model's Q, we think it is real Q)
        self.target_net.eval()
        q_next = self.target_net(torch.tensor(batch_memory[:,-self.n_features:],dtype=torch.float)).detach().numpy()
        
        # use eval_net get q_eval (new model's Q, we think it is prediction Q)
        self.eval_net.eval()
        q_eval = self.eval_net(torch.tensor(batch_memory[:,:self.n_features],dtype=torch.float)).detach().numpy()

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # forward
        self.eval_net.train()
        q_eval = self.eval_net(torch.tensor(batch_memory[:,:self.n_features],dtype=torch.float))
        
        # loss
        self.cost = self.loss(q_eval,torch.tensor(q_target,dtype=torch.float))
        #print(self.cost)
        
        # backward
        self.optimizer.zero_grad()
        self.cost.backward()
        self.optimizer.step()
    
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib as mpl
        mpl.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

EPISODE = 50
        
if __name__ == "__main__":
    env = gym.make('CartPole-v0')
#     env = gym.make('MountainCar-v0')
    env = env.unwrapped

    print(env.action_space)
    print(env.observation_space)
    print(env.observation_space.high)
    print(env.observation_space.low)

    RL = DeepQNetwork(n_actions=env.action_space.n,
                      n_features=env.observation_space.shape[0],
                      learning_rate=0.01, 
                      e_greedy=0.9,
                      replace_target_iter=100, 
                      memory_size=2000,
                      e_greedy_increment=0.001,
                      output_graph=True
                     )

    total_steps = 0

    for i_episode in range(EPISODE):
        observation = env.reset()
        ep_r = 0
        while True:
            env.render()

            action = RL.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            # the smaller theta and closer to center the better
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            reward = r1 + r2

            RL.store_transition(observation, action, reward, observation_)

            ep_r += reward
            if total_steps > 1000:
                RL.learn()

            if done:
                print('episode: ', i_episode,
                      'ep_r: ', round(ep_r, 2),
                      ' epsilon: ', round(RL.epsilon, 2))
                break

            observation = observation_
            total_steps += 1

    RL.plot_cost()
