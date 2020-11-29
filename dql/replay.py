#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:03:54 2020

@author: haotian teng
"""
import numpy as np
import torch
import random


class ReplayMemory(object):
    def __init__(self, config):
        self.capacity = config.BUFFER_SIZE
        self.batch_size = config.BATCH_SIZE
        self.position = 0
        self.count = 0
        self.history_length = config.FRAME_STACK
        #Preallocate Memory to ensure the RAM has enough capacity
        self.images = np.empty((self.capacity,config.IMAGE_SIZE[0],config.IMAGE_SIZE[1]),dtype = config.IMG_DTYPE)
        self.actions = np.empty(self.capacity,dtype = np.uint8)
        self.rewards = np.empty(self.capacity,dtype = np.int)
        self.done = np.empty(self.capacity,dtype = np.bool)
        self.state = np.empty((self.batch_size,config.FRAME_STACK)+config.IMAGE_SIZE,dtype = np.float32)
        self.next_state = np.empty((self.batch_size,config.FRAME_STACK)+config.IMAGE_SIZE,dtype = np.float32)
        
    def push(self, image, action, reward, done):
        """Saves a transition."""
        self.images[self.position] = image
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.done[self.position] = done
        self.count = max(self.count,self.position+1)
        self.position = (self.position+1) % self.capacity
        
    def get_state(self,index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        index = (index%self.count)
        if index >= self.history_length-1:    
            return self.images[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            indexes = self._circular_index(index + 1 - self.history_length,index+1)
            return self.images[indexes, ...]
    
    def _circular_index(self,start,end):
        return np.arange(start,end)%self.count
    
    def sample(self):
        assert self.count > self.history_length
        indexes = []
        while len(indexes) < self.batch_size:
            while True:
                index = random.randint(0, self.count - 1)
                # if wraps over current pointer, then get new one
                if self.position in self._circular_index(index - self.history_length,index+1):
                    continue
                # if wraps over episode end, then get new one
                if self.done[self._circular_index(index - self.history_length,index)].any():
                    continue
                # otherwise use this index
                break
          
            # NB! having index first is fastest in C-order matrices
            self.state[len(indexes), ...] = self.get_state(index-1)
            self.next_state[len(indexes), ...] = self.get_state(index)
            indexes.append(index)
    
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        done = self.done[indexes]
        return (torch.from_numpy(self.state),
                torch.from_numpy(self.next_state),
                torch.from_numpy(actions),
                torch.from_numpy(rewards),
                torch.from_numpy(done))
       
    def __len__(self):
        return self.count
    
    def __getitem__(self,idx):
        return  (self.get_state(idx-1),
                self.get_state(idx),
                self.actions[idx],
                self.rewards[idx],
                self.done[idx])