#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:03:54 2020

@author: haotian teng
"""
import gym
import random
import numpy as np
import cv2

def preprocess(img,dtype = np.float32):
#    img_gray = np.mean(img, axis=2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_down = cv2.resize(img_gray,(84,84),interpolation=cv2.INTER_AREA)
    img_norm = img_down/255.
    img_norm = np.asarray(img_norm,dtype = dtype)
    return img_norm          

class Enviroment(object):
    def __init__(self,config):
        self.env = gym.make(config.ENV_NAME)
        self.random_start = config.RANDOM_START
        self.dtype = config.IMG_DTYPE
    def new_game(self):
        frame = self.env.reset()
        for _ in np.arange(random.randint(0,self.random_start)):
            frame, reward, done, info = self.env.step(0)
        return preprocess(frame,self.dtype)
    
    def step(self,action):
        frame, reward, done, info = self.env.step(action)
        return preprocess(frame,self.dtype),reward,done,info
    
    def random_step(self):
        action = self.env.action_space.sample()
        frame, reward, done, info = self.step(action)
        return frame,action,reward,done,info