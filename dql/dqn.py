#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 04:48:51 2020

"""

import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import animation
import matplotlib.pyplot as plt
from tqdm import tqdm
from dql.replay import ReplayMemory
from dql.environment import Enviroment
from dql.utils import tp_order
import json
### Import for testing should be removed to main after
#from dqn.environment import GymEnvironment    
class DQN(nn.Module):

    def __init__(self, config): 
        super(DQN, self).__init__()
        self.config = config
        self.device = config.DEVICE
        self.blocks = config.BLOCK_CONFIG
        self.access_order = tp_order(self.blocks)
        # an affine operation: y = Wx + b
        self.build_net()
    
    def check_config(self):
        # Check block configuration to make sure the parents and childs are 
        # consistent and channel number is correct.
        raise NotImplementedError
    
    def build_net(self):
        self.layers = [None]*len(self.blocks)
        self.out_blocks = [] #The id of final output blocks
        self.input_blocks = [] #The id of initial input blocks
        block_n = len(self.blocks)
        self.shapes = np.zeros((block_n,2),dtype = np.int)
        for block_id in self.access_order:
            block = self.blocks[block_id]
            if not block['parents']:
                #If the current block has no parent, feed the image stack.
                layer = nn.Conv2d(in_channels = self.config.FRAME_STACK,
                                         out_channels = block['out_channels'],
                                         kernel_size = block['kernel_size'],
                                         stride = block['stride'],
                                         padding = 0)
                self.shapes[block_id] = self.get_shape(self.config.IMAGE_SIZE,layer)
                self.input_blocks.append(block_id)
            else:
                layer = nn.Conv2d(in_channels = block['in_channels'],
                         out_channels = block['out_channels'],
                         kernel_size = block['kernel_size'],
                         stride = block['stride'],
                         padding = 0)
                self.shapes[block_id] = self.get_shape(np.max(self.shapes[block['parents']],axis = 0),layer)
            super().__setattr__('conv'+str(block_id),layer)
            if not block['childs']:
                #If the block has no child block, gather its output.
                self.out_blocks.append(block_id)
            self.layers[block_id] = layer
        out_shape = np.max(self.shapes[self.out_blocks],axis=0)
        out_channel = np.sum([self.layers[out_block].out_channels for out_block in self.out_blocks])
        self.fc1 = nn.Linear(out_shape[0]*out_shape[1]*out_channel,self.config.HIDDEN_SIZE)
        self.out = nn.Linear(self.config.HIDDEN_SIZE, self.config.OUT_SIZE)
        
    def get_shape(self,in_size,layer):
        padding = np.array(layer.padding)
        dilation = np.array(layer.dilation)
        kernal_size = np.array(layer.kernel_size)
        stride = np.array(layer.stride)
        shape = (in_size+2*padding-dilation*(kernal_size-1)-1)/stride+1
        return shape.astype(np.int)
    
    def _pad_cat(self,x,y):
        #Pad the two variables into same heigth and width before concatenating
        if x.shape[-2:] == y.shape[-2:]:
            return torch.cat([x,y],dim = 1)
        else:
            max_shape = np.maximum(np.array(x.shape[-2:]),y.shape[-2:])
            return torch.cat([self._pad_to(x,max_shape),self._pad_to(y,max_shape)],dim = 1)
            
    def _pad_to(self,x,max_shape):
        x_shape = x.shape[-2:]
        if (x_shape == max_shape).all():
            return x
        else:
            delta_shape = np.array(max_shape,dtype = np.int) - np.array(x_shape,dtype = np.int)
            return F.pad(x,(np.int(delta_shape[1]/2),delta_shape[1]-np.int(delta_shape[1]/2),
                            np.int(delta_shape[0]/2),delta_shape[0]-np.int(delta_shape[0]/2)))
        
    def forward(self, x):
        #Link the blocks according to the configuration
        block_n = len(self.blocks)
        input_cache = [None]*block_n
        for i in self.input_blocks:
            input_cache[i] = x
        x = None
        output = None
        for idx in self.access_order:
            x = F.relu(self.layers[idx](input_cache[idx]))
            if idx in self.out_blocks:
                output = x if output is None else self._pad_cat(output,x)
            else:
                for child_id in self.blocks[idx]['childs']:
                    if input_cache[child_id] is None:
                        input_cache[child_id] = x
                    else:
                        input_cache[child_id] = self._pad_cat(input_cache[child_id],x)
            input_cache[idx] = None
        x = output
        x = x.view(x.size()[0], -1)  # In: (, 32, 13, 10) Out: (, 4160)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


class History(object):
    def __init__(self,config):
        self.stack_n = config.FRAME_STACK
        self.buffer = np.zeros((self.stack_n,config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]),dtype = config.IMG_DTYPE)
    
    def add(self,image):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1,:] = image
        
    def get(self):
        return torch.from_numpy(self.buffer.astype(np.float32))


def init_memory(env,buffer,initial_size,history_length):
    frame = env.new_game()
    for _ in np.arange(history_length):
        buffer.push(frame,0,0,False)
    for _ in tqdm(np.arange(initial_size)):
        frame,action,reward,done,info = env.random_step()
        buffer.push(frame,action,reward,done)
        if done:
            frame = env.new_game()
            for _ in np.arange(history_length):
                buffer.push(frame,0,0,False)



def save_frames_as_gif(frames, path='./', filename='pong_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 50.0, frames[0].shape[0] / 50.0), dpi=144)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=0)
    anim.save(path + filename, writer='imagemagick', fps=12)
    
def train(conf_file = None):
    def train_step():
        if len(memory) < config.BATCH_SIZE:
            return None
        
        batch_state,batch_next_state,batch_action,batch_reward,batch_done = memory.sample()
        batch_state = batch_state.to(config.DEVICE)
        batch_next_state = batch_next_state.to(config.DEVICE)
        batch_action = batch_action.to(config.DEVICE)
        batch_reward = batch_reward.to(config.DEVICE)
        batch_done = batch_done.to(config.DEVICE)
        current_Q = Q(batch_state).gather(1, batch_action.unsqueeze(1).long())
    
        expected_Q = batch_reward.float()
        expected_Q[~batch_done] += config.GAMMA * target_Q(batch_next_state[~batch_done]).max(1)[0].detach()
    
        loss = F.mse_loss(current_Q, expected_Q.unsqueeze(1))
        #loss = F.smooth_l1_loss(current_Q, current_Q.unsqueeze(1))
        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
    
        for param in Q.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        return loss.item(),current_Q.mean().item()
    
    def predict(state,eps):
        if np.random.rand() < eps:
            return env.env.action_space.sample()
        q_vals = Q(state.to(config.DEVICE).unsqueeze(0)).squeeze()
        return q_vals.argmax().item()
    class ENVIROMENT_CONFIG(object):
        ENV_NAME = 'PongDeterministic-v4'
        RANDOM_START = 0
        IMG_DTYPE = np.float32
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class NN_CONFIG(ENVIROMENT_CONFIG):
        HIDDEN_SIZE = 256
        BLOCK_CONFIG = [{'parents':[],
                         'childs':[1,2],
                         'in_channels':None,
                         'out_channels':16,
                         'kernel_size':[5,4],
                         'stride':2},
                        {'parents':[0],
                         'childs':[2],
                         'in_channels':16,
                         'out_channels':32,
                         'kernel_size':3,
                         'stride':1},
                         {'parents':[0,1],
                         'childs':[3],
                         'in_channels':48,
                         'out_channels':32,
                         'kernel_size':3,
                         'stride':1},
                          {'parents':[2],
                         'childs':[],
                         'in_channels':32,
                         'out_channels':32,
                         'kernel_size':3,
                         'stride':1}]
        
    class DQN_CONFIG(NN_CONFIG):
        BASE = 1000
        BUFFER_SIZE = 200 * BASE 
        BATCH_SIZE = 32
        IMAGE_SIZE = (84,84)
        GAMMA = 1.0
        T_MAX = 5000
        EPISODE_MAX = 1000
        TARGET_UPDATE = 1*BASE
        EPS_0 = 1.0
        EPS_MIN = 0.1
        EPS_LEN = 2*BUFFER_SIZE
        INITIAL_COLLECTION=5 * BASE
        REPEAT_ACTIONS = 1
        FRAME_STACK = 4
        LEARNING_RATE = 1e-4
        SAVE_LATEST = 5
        
    config = DQN_CONFIG
    if conf_file is not None:
        with open(conf_file,'r') as f:
            config.BLOCK_CONFIG = json.load(f)
    train_hist = []
    env = Enviroment(config)
    #device = torch.device('cpu')
    config.OUT_SIZE = env.env.action_space.n
    Q = DQN(config).to(config.DEVICE)
    ####### load model ##########
    #Q.load_state_dict(torch.load('pong_Q'))
    
    target_Q = DQN(config).to(config.DEVICE)
    target_Q.load_state_dict(Q.state_dict())
    target_Q.eval()
    memory = ReplayMemory(config)
    optimizer = optim.Adam(Q.parameters(),lr = config.LEARNING_RATE)    
    
    
    global_step = 0
    print("Begin initial replay memory collection.\n")
    init_memory(env,memory,config.INITIAL_COLLECTION,config.FRAME_STACK)
    frame_buffer = History(config)
    save_list = []
    print("Begin training.")
    for i_episode in range(config.EPISODE_MAX):
        tot_reward = 0
        frame = env.new_game()
        for _ in np.arange(config.FRAME_STACK):
            frame_buffer.add(frame)
            memory.push(frame,0,0,False)
        t_start = time.time()
        eps = max(config.EPS_MIN, config.EPS_0*(config.EPS_LEN-global_step)/config.EPS_LEN)
        for t in range(config.T_MAX):
            global_step+=1
            state = frame_buffer.get()
            action = predict(state,eps)
            cumulative_reward = 0
            for i in np.arange(config.REPEAT_ACTIONS):    
                frame, reward, done, info = env.step(action)
                cumulative_reward += reward
                if done:
                    break
            frame_buffer.add(frame)
            memory.push(frame, 
                        action,
                        cumulative_reward,
                        done)
            tot_reward += cumulative_reward
    #            raise
            loss,q_val = train_step()
            if global_step % config.TARGET_UPDATE == 0:
                target_Q.load_state_dict(Q.state_dict())
                torch.save(Q.state_dict(), 'pong_Q%d'%(global_step))
                torch.save(target_Q.state_dict(), 'pong_Q_target_%d'%(global_step))
                save_list.append(('pong_Q%d'%(global_step),'pong_Q_target_%d'%(global_step)))
                if len(save_list) > config.SAVE_LATEST:
                    [os.remove(x) for x in save_list.pop(0)]
            if done:
                break  
                    
        train_hist += [tot_reward]
        print("Epoch:%d Global step:%d Loss:%.5f Q value: %.5f Total Reward:%.0f Trail Length:%d Epsilon:%.2F Elapsed Time:%.2f Buffer size:%d"%(i_episode, global_step, loss, q_val, tot_reward, t+1, eps, time.time() - t_start, len(memory)))
    ###
    
    plt.figure(figsize = (10,10))
    plt.plot(train_hist)
    #    plt.plot(np.arange(0,EPISODE_MAX,10),
    #             np.array(train_hist).reshape(-1, EPISODE_MAX).mean(axis = 1))
    plt.xlabel('# of Episode', fontsize = 20)
    plt.ylabel('Total Reward', fontsize = 20)
    
    
    def simulate(env, horizon,config, render = False):
        tot_reward = 0
        frame = env.new_game()
        frame_buffer = History(config)
        for _ in np.arange(config.FRAME_STACK):
            frame_buffer.add(frame)
        movie_frame = []
        for t in range(horizon):
            if render:
                #env.render()
                env.env.render()
                movie_frame.append(env.env.render(mode="rgb_array"))
                time.sleep(1/24)
                
            state = frame_buffer.get()
            action = predict(state,eps = 0)
            frame, reward, done, info = env.step(action)
            frame_buffer.add(frame)
            tot_reward += reward
            if done:
                break
                
        if render:    
            env.env.close()
                
        return tot_reward, reward, done, t, movie_frame
    
    ###
    env.random_start = 0
    _ = simulate(env, 100,config, True)
    torch.save(Q.state_dict(), 'pong_Q_final')
    torch.save(target_Q.state_dict(), 'pong_Q_target_final')
    
    Q = DQN(config).to(config.DEVICE)
    target_Q = DQN(config).to(config.DEVICE)
    ####### load model ##########
    
    Q.load_state_dict(torch.load('pong_Q_final'))
    target_Q.load_state_dict(torch.load('pong_Q_target_final'))
    
    reward_tot, reward, t, done, frames = simulate(env, 500,config, True)
    save_frames_as_gif(frames[::2])
    return reward_tot
if __name__ == "__main__":
    train("/home/heavens/twilight/NASRL_DQN/configs/0/2.json")


