import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

EPISODE = 10000

SLOW_EPISODE = 8000

np.random.seed(1)

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell 1
        hell1_center = origin + np.array([UNIT * (MAZE_H-2), UNIT * (MAZE_W-2)])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        
        # hell 2
        hell2_center = origin + np.array([UNIT * (MAZE_H-1), UNIT * (MAZE_W-3)])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + np.array([UNIT * (MAZE_H-1), UNIT * (MAZE_W-2)])
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self,episode):
        self.update()
        if episode > SLOW_EPISODE:
            time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        next_coords = self.canvas.coords(self.rect)  # next state
        
        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 1
            done = True
        elif next_coords in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
        else:
            reward = 0
            done = False
        
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2]))/(MAZE_H*UNIT)
        return s_, reward, done

    def render(self,episode):
        if episode > SLOW_EPISODE:
            time.sleep(0.1)
            self.update()
        else:
            self.update() 

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
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter("logs/")
                self.tb_writer.add_graph(self.eval_net, input_to_model=None, verbose=False)
            except:
                pass

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
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

def run_maze():
    step = 0
    for episode in range(EPISODE):
        print("episode:",episode)
        
        # initial observation
        observation = env.reset(episode)

        while True:
            # fresh env
            env.render(episode)

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                env.render(episode)
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # e_greedy_increment=True,
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()
