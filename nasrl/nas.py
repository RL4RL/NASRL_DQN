#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 17:50:06 2020

"""
import torch
import torch.nn as nn
import numpy as np
import copy
import torch.optim as optim
from dql import dqn
import os
import json
import argparse
import sys
class RNN(nn.Module):
    def __init__(self,config):
        super(RNN, self).__init__()
        self.config = config
        self.lstm = nn.LSTM(input_size = config.HIDDEN_SIZE,
                            hidden_size = config.HIDDEN_SIZE,
                            num_layers = config.LAYER_N)
        self.embedding = nn.Embedding(config.OUT_SIZE,config.HIDDEN_SIZE)
        self.anchor_embedding = nn.Linear(in_features = config.MAX_ROLLOUT-1,
                                        out_features = config.HIDDEN_SIZE)
        self.classifiers = []
        self.block_n = len(self.config.STATE_SIZE)
        for i in np.arange(len(config.STATE_SIZE)-1):
            classifier = nn.Linear(in_features=config.HIDDEN_SIZE,
                                   out_features=config.STATE_SIZE[i])
            self.classifiers.append(classifier)
        self.wprev = nn.Linear(in_features=config.HIDDEN_SIZE,
                               out_features=config.ANCHOR_SIZE)
        self.wcurr = nn.Linear(in_features=config.HIDDEN_SIZE,
                               out_features=config.ANCHOR_SIZE)
        self.v = torch.randn(config.ANCHOR_SIZE)
    def forward(self,batch_size):
        anchor_hs = [torch.zeros(batch_size,self.config.HIDDEN_SIZE)]
        input = torch.randn(1,batch_size,self.config.HIDDEN_SIZE)
        h = torch.zeros(self.config.LAYER_N,batch_size,self.config.HIDDEN_SIZE)
        c = torch.zeros(self.config.LAYER_N,batch_size,self.config.HIDDEN_SIZE)
        self.outputs_prob = []
        self.logits = []
        for i in np.arange(self.config.MAX_ROLLOUT*len(self.config.STATE_SIZE)):
            state_idx = i%self.block_n
            cycle_idx = i//self.block_n
            if state_idx < self.block_n-1:
                output_h,(h,c) = self.lstm(input,(h,c))
                logit = self.classifiers[state_idx](output_h)
                self.logits.append(logit)
                output_p = nn.functional.softmax(logit,dim=-1)
                output_p = torch.squeeze(output_p)
                self.outputs_prob.append(output_p)
            else:
                output_h,(h,c) = self.lstm(input,(h,c))
                output_h = torch.squeeze(output_h)
                anchor_hs.append(output_h)
                anchor_o = []
                for i_prev in np.arange(cycle_idx):
                    p = torch.sum(self.v*nn.functional.tanh(self.wprev(anchor_hs[i_prev])+self.wcurr(output_h)),dim = -1)
                    anchor_o.append(torch.sigmoid(p))
                if not anchor_o:
                    anchor_o = torch.zeros((batch_size,self.config.MAX_ROLLOUT-1))
                else:
                    anchor_o = torch.stack(anchor_o,dim = 1)
                    anchor_o = nn.functional.pad(anchor_o,(0,self.config.MAX_ROLLOUT-cycle_idx-1),"constant",value = 0)
                self.outputs_prob.append(anchor_o)
                input = self.anchor_embedding(anchor_o)
                input = torch.unsqueeze(input,dim = 0)
        return self.outputs_prob
    
    def loss(self,rewards,ys):
        rewards = torch.tensor(rewards)
        for i,output_prob in enumerate(self.outputs_prob):
            y = torch.tensor(ys[i])
            if i==0:
                loss = torch.dot(rewards,torch.sum(y*torch.log(output_prob),dim = 1))
            else:
                loss += torch.dot(rewards,torch.sum(y*torch.log(output_prob),dim = 1))
        return loss                
class Blocks(object):
    def __init__(self,config):
        self.config = config
        self.structure_dict = config.STRUCTURE_DICT
        self.fh = self.structure_dict["FILTER_H"]
        self.fw = self.structure_dict["FILTER_W"]
        self.fn = self.structure_dict["FILTER_N"]
        self.sh = self.structure_dict["STRIDE_H"]
        self.sw = self.structure_dict["STRIDE_W"]
        self.block_proto = config.BLOCK_PROTO
        self.state_n = len(config.STATE_SIZE)
        self.block_n = config.MAX_ROLLOUT
    def gen_conf(self,rnn_outputs):
        """Given an output of block indexs, sample a configuration.
        Args:
            outputs: A list of numpy array with length MAX_ROLLOUT*STATE_N
        """
        if torch.is_tensor(rnn_outputs[0]):
            rnn_outputs = [x.cpu().detach().numpy() for x in rnn_outputs]
        batch_size = rnn_outputs[0].shape[0]
        roll_n = len(rnn_outputs)
        assert roll_n%self.state_n == 0
        archtectures = []
        for i in np.arange(batch_size):
            archtectures.append([x[i] for x in rnn_outputs]) #Rearrange the outputs to [batch_size,max_rollout*state_n]
        confs = []
        y = []
        for batch_i,arche in enumerate(archtectures):
            configure = []
            samples = []
            for i in np.arange(0,len(arche),self.state_n):
                block,sample = self._gen_block(arche[i:i+self.state_n])
                configure.append(block)
                samples += sample
            self._link_blocks(configure)
            confs.append(configure)
            y.append(samples)
        y = self.onehot_encode(y)
        return confs,y
    
    def _gen_block(self,states):
        block = copy.deepcopy(self.block_proto)
        sample = []
        for state in states[:-1]:
            sample.append(np.random.choice(np.arange(len(state)),
                                           p = state))
        block['kernel_size'] = [self.fw[sample[0]],self.fh[sample[1]]]
        block['out_channels'] = self.fn[sample[2]]
        block['stride'] = [self.sw[sample[3]],self.sh[sample[4]]]
        parents = np.where(np.random.random(self.block_n-1)<states[5])[0]
        sample.append(parents)
        block['parents'] = [int(x) for x in parents]
        return block,sample
    
    def onehot_encode(self,samples):
        y = []
        for i in np.arange(self.block_n*self.state_n):
            onehot = [self.onehot_vector(x[i],self.config.STATE_SIZE[i%self.state_n]) for x in samples]
            y.append(np.stack(onehot))
        return y
    
    def onehot_vector(self,x,size):
        if type(x) is not list:
            onehot = np.zeros(size)
            onehot[x] = 1
        else:
            onehot = np.zeors((len(x),size))
            onehot[np.arange(len(x)),x] = 1
        return onehot
        
    def _link_blocks(self,blocks):
        for block_i,block in enumerate(blocks):
            for parent in block['parents']:
                blocks[parent]['childs'].append(block_i)
            if block['parents']:
                block['in_channels'] = int(np.sum([blocks[x]['out_channels'] for x in block['parents']]))
    

class Trainer(object):
    #TODO wrap the training process in a trainer.
    def __init__(self,net):
        pass
    
    def load(self,save_f):
        pass
    
    def save(self):
        pass
    
    def loss(self,x,r):
        pass
    
    def train_step(self,x,r):
        pass
    
    def train(self):
        pass
    
def train(cmd_args):
    class NAS_CONFIG(object):
        STRUCTURE_DICT = {"FILTER_H":[1,3,5],
                          "FILTER_W":[1,3,5],
                          "FILTER_N":[24,48],
                          "STRIDE_H":[1,2],
                          "STRIDE_W":[1,2]}
        MAX_ROLLOUT=4
        HIDDEN_SIZE = 35
        STATE_SIZE = [len(x) for x in STRUCTURE_DICT.values()]+[MAX_ROLLOUT-1]
        #The anchor points can be at most MAX_ROLLOUT+1, 0 means no parent.
        OUT_SIZE = sum(STATE_SIZE[:-1])
        ANCHOR_SIZE = 10
        LAYER_N = 2
        LR = 0.006
    class BLOCKS_CONFIG(NAS_CONFIG):
        BLOCK_PROTO = {'parents':[],
                      'childs':[],
                      'in_channels':None,
                      'out_channels':None,
                      'kernel_size':None,
                      'stride':None}
    class TRAINING_CONFIG(BLOCKS_CONFIG):
        CONF_FOLDER = cmd_args.conf_f
        TRAIN_STEP = 10
        BATCH_SIZE = cmd_args.batch_size
        LEARNING_RATE = 1e-3
        
    def train_step(reward,y):
        optimizer.zero_grad()
        loss = net.loss(reward,y)
        loss.backward()
        optimizer.step()
        return loss.item()
        
        
    config = TRAINING_CONFIG()
    net = RNN(config)
    optimizer = optim.Adam(net.parameters(),lr = config.LEARNING_RATE)  
    blocks = Blocks(config)
    reward_baseline = 0
    decay = 0.9
    if not os.path.isdir(config.CONF_FOLDER):
        os.mkdir(config.CONF_FOLDER)
    for step in np.arange(config.TRAIN_STEP):
        outputs = net.forward(config.BATCH_SIZE)
        confs,y = blocks.gen_conf(outputs)
        n_f = os.path.join(config.CONF_FOLDER,str(step))
        if not os.path.isdir(n_f):
            os.mkdir(n_f)
        rewards = []
        conf_fs = []
        for conf_i,conf in enumerate(confs):
            conf_f = os.path.join(n_f,"%d.json"%(conf_i))
            conf_fs.append(conf_f)
            with open(conf_f,'w+') as f:
                json.dump(conf,f)
        for conf_f in conf_fs:
            rewards.append(dqn.train(conf_f,os.path.splitext(conf_f)[0]))
            with open(conf_f+".reward","w+") as f:
                f.write(str(rewards[-1]))
        rewards = np.array(rewards)
        reward_baseline = reward_baseline*decay + np.mean(rewards)*(1-decay)
        rewards = rewards-reward_baseline
        train_step(rewards,y)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='nas',
                                     description='Neural Architecture Search.')
    parser.add_argument('-c', '--conf_f', required = True,
                        help="Configuration folder.")
    parser.add_argument('-b', '--batch_size', default = 3,
                        help="Batch size.",type = int)
    args = parser.parse_args(sys.argv[1:])
    train(args)
