"""
My view:
step 1. Like Sarsa, we use Q table to store the Q value of every <state,action> tuple, so state_num equals to row_num and action num equals to collum_num;
step 2. Like Sarsa, when we choose action for current_state, we have e_greedy probability to choose the current_action which has max Q value of this row(current_state), and we also have (1-e_greedy) probability to random choose action;
step 3. Like Sarsa, we do the current_action which is choosed by (step 2), and we will get the reward and the next_state, so we will get the <current_state,current_action,reward,next_state> tuple;
step 4. Like Sarsa, we can choose next_action for next_state;
step 5. Like Sarsa, Sarsa calculate q_target(=reward + gamma*q_table[next_state,next_value]);
step 6. Like Sarsa, we use Q table to get q_prediciton(q_table[current_state,current_action]);
step 7. Unlike Sarsa, we use E table(Eligibility_trace table) to get the access frequency(E value) for every <state,action> tuple, but we will discount the E value by multiply gamma*lambda on every step;
step 8. Unlike Sarsa, we update the Q table: q_predictin_new={q_prediction_old+learning_rate*(q_target-q_prediction_old)*E[state,action]}

Attention:
1. The most important thing is to know the difference of Sarsa and SarsaLambda---when update q_prediction, SarsaLambda need multiply E value.
"""


import numpy as np
import pandas as pd
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

UNIT = 40   # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width

EPISODE = 1000

SLOW_EPISODE = 800

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
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
        return self.canvas.coords(self.rect)

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

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 10
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:
            reward = -1
            done = True
            s_ = 'terminal'
            
        else:
            reward = 0
            done = False

        return s_, reward, done

    def render(self,episode):
        if episode > SLOW_EPISODE:
            time.sleep(0.1)
            self.update()
        else:
            self.update() 

class SarsaLambdaTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,trace_decay=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        
        # Sarsa Lambda
        self.lambda_ = trace_decay
        self.eligibility_trace = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
            
        # Sarsa Lambda
        delta = q_target - q_predict
        
        # method 1
        # self.eligibility_trace.loc[s, a] += 1
        
        # method 2
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1
        
        self.q_table += self.lr * delta * self.eligibility_trace
        
        self.eligibility_trace *= self.gamma*self.lambda_

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            
            # Sarsa Lambda
            self.eligibility_trace = self.eligibility_trace.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.eligibility_trace.columns,
                    name=state,
                )
            )
    
def update():
    for episode in range(EPISODE):
        
        print("episode:",episode)
        
        # initial observation
        observation = env.reset(episode)
    
        # RL choose action based on observation
        action = RL.choose_action(str(observation))

        while True:
            # fresh env
            env.render(episode)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            # RL choose action_ based on observation_
            action_ = RL.choose_action(str(observation_))

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_),action_)

            # swap observation
            observation = observation_
            
            action = action_

            # break while loop when end of this episode
            if done:
                env.render(episode)
                break
    
    # end of game
    print('game over')
    env.destroy()

env = Maze()
RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

env.after(100, update)
env.mainloop()
