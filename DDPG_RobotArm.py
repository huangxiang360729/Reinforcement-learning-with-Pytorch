import pyglet
import numpy as np

import torch
from torch import nn
from torch.nn import init

from copy import deepcopy

MAX_EPISODES = 900
MAX_EP_STEPS = 200
ON_TRAIN = True

LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 30000
BATCH_SIZE = 32

###############################  DDPG  ####################################
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = torch.tensor(action_bound[1],dtype=torch.float)
        
        num_hiddens = 300
        
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
        
        num_hiddens = 300
        
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
        # "t = (1 - TAU)* t + TAU * e" is error, very important!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for t,e in zip(self.actor_target_net.parameters(),self.actor_eval_net.parameters()):
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
        
        tensor_batch_next_actions = self.actor_target_net(tensor_batch_next_states)
        tensor_Q_next = self.critic_target_net(tensor_batch_next_states, tensor_batch_next_actions)
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
        
    def save(self):
        torch.save(self.actor_eval_net,"actor_eval_net.pth")

    def restore(self):
        self.actor_eval_net = torch.load("actor_eval_net.pth")

class ArmEnv(object):
    viewer = None
    dt = .1    # refresh rate
    action_bound = [-1, 1]
    goal = {'x': 100., 'y': 100., 'l': 40}
    state_dim = 9
    action_dim = 2

    def __init__(self):
        self.arm_info = np.zeros(
            2, dtype=[('l', np.float32), ('r', np.float32)])
        self.arm_info['l'] = 100        # 2 arms length
        self.arm_info['r'] = np.pi/6    # 2 angles information
        self.on_goal = 0

    def step(self, action):
        done = False
        action = np.clip(action, *self.action_bound)
        self.arm_info['r'] += action * self.dt
        self.arm_info['r'] %= np.pi * 2    # normalize

        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])    # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0]) / 400, (self.goal['y'] - a1xy_[1]) / 400]
        dist2 = [(self.goal['x'] - finger[0]) / 400, (self.goal['y'] - finger[1]) / 400]
        r = -np.sqrt(dist2[0]**2+dist2[1]**2)

        # done and reward
        if self.goal['x'] - self.goal['l']/2 < finger[0] < self.goal['x'] + self.goal['l']/2:
            if self.goal['y'] - self.goal['l']/2 < finger[1] < self.goal['y'] + self.goal['l']/2:
                r += 1.
                self.on_goal += 1
                if self.on_goal > 50:
                    done = True
        else:
            self.on_goal = 0

        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand()*400.
        self.goal['y'] = np.random.rand()*400.
        self.arm_info['r'] = 2 * np.pi * np.random.rand(2)
        self.on_goal = 0
        (a1l, a2l) = self.arm_info['l']  # radius, arm length
        (a1r, a2r) = self.arm_info['r']  # radian, angle
        a1xy = np.array([200., 200.])  # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
        finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_  # a2 end (x2, y2)
        # normalize features
        dist1 = [(self.goal['x'] - a1xy_[0])/400, (self.goal['y'] - a1xy_[1])/400]
        dist2 = [(self.goal['x'] - finger[0])/400, (self.goal['y'] - finger[1])/400]
        # state
        s = np.concatenate((a1xy_/200, finger/200, dist1 + dist2, [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.arm_info, self.goal)
        self.viewer.render()

    def sample_action(self):
        return np.random.rand(2)-0.5    # two radians


class Viewer(pyglet.window.Window):
    bar_thc = 5

    def __init__(self, arm_info, goal):
        # vsync=False to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=400, height=400, resizable=False, caption='Arm', vsync=False)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.arm_info = arm_info
        self.goal_info = goal
        self.center_coord = np.array([200, 200])

        self.batch = pyglet.graphics.Batch()    # display whole batch at once
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,    # 4 corners
            ('v2f', [goal['x'] - goal['l'] / 2, goal['y'] - goal['l'] / 2,                # location
                     goal['x'] - goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] + goal['l'] / 2,
                     goal['x'] + goal['l'] / 2, goal['y'] - goal['l'] / 2]),
            ('c3B', (86, 109, 249) * 4))    # color
        self.arm1 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [250, 250,                # location
                     250, 300,
                     260, 300,
                     260, 250]),
            ('c3B', (249, 86, 86) * 4,))    # color
        self.arm2 = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', [100, 150,              # location
                     100, 160,
                     200, 160,
                     200, 150]), ('c3B', (249, 86, 86) * 4,))

    def render(self):
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def _update_arm(self):
        # update goal
        self.goal.vertices = (
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] - self.goal_info['l']/2,
            self.goal_info['x'] + self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2,
            self.goal_info['x'] - self.goal_info['l']/2, self.goal_info['y'] + self.goal_info['l']/2)

        # update arm
        (a1l, a2l) = self.arm_info['l']     # radius, arm length
        (a1r, a2r) = self.arm_info['r']     # radian, angle
        a1xy = self.center_coord            # a1 start (x0, y0)
        a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy   # a1 end and a2 start (x1, y1)
        a2xy_ = np.array([np.cos(a1r+a2r), np.sin(a1r+a2r)]) * a2l + a1xy_  # a2 end (x2, y2)

        a1tr, a2tr = np.pi / 2 - self.arm_info['r'][0], np.pi / 2 - self.arm_info['r'].sum()
        xy01 = a1xy + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc
        xy02 = a1xy + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy11 = a1xy_ + np.array([np.cos(a1tr), -np.sin(a1tr)]) * self.bar_thc
        xy12 = a1xy_ + np.array([-np.cos(a1tr), np.sin(a1tr)]) * self.bar_thc

        xy11_ = a1xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc
        xy12_ = a1xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy21 = a2xy_ + np.array([-np.cos(a2tr), np.sin(a2tr)]) * self.bar_thc
        xy22 = a2xy_ + np.array([np.cos(a2tr), -np.sin(a2tr)]) * self.bar_thc

        self.arm1.vertices = np.concatenate((xy01, xy02, xy11, xy12))
        self.arm2.vertices = np.concatenate((xy11_, xy12_, xy21, xy22))

    # convert the mouse coordinate to goal's coordinate
    def on_mouse_motion(self, x, y, dx, dy):
        self.goal_info['x'] = x
        self.goal_info['y'] = y
        
# set env
env = ArmEnv()
s_dim = env.state_dim
a_dim = env.action_dim
a_bound = env.action_bound

# set RL method (continuous)
rl = DDPG(a_dim, s_dim, a_bound)

steps = []
def train():
    # start training
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_r = 0.
        for j in range(MAX_EP_STEPS):
            # env.render()

            a = rl.choose_action(s)

            s_, r, done = env.step(a)

            rl.store_transition(s, a, r, s_)

            ep_r += r
            if rl.pointer > MEMORY_CAPACITY:
                # start to learn once has fulfilled the memory
                rl.learn()

            s = s_
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save()


def eval():
    rl.restore()
    env.render()
    env.viewer.set_vsync(True)
    s = env.reset()
    while True:
        env.render()
        a = rl.choose_action(s)
        s, r, done = env.step(a)


if ON_TRAIN:
    train()
else:
    eval()
