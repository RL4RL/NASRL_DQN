#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:59:31 2020

@author: haotian teng
"""
import numpy as np
def tp_order(nodes):
    #Return a topological order of given nodes
    indegrees = np.array([len(x['parents']) for x in nodes])
    order = []
    while len(order)<len(nodes):
        assert(np.sum(indegrees==0)>0)
        for node in np.where(indegrees==0)[0]:
            order.append(node)
            indegrees[node] = -1
            for child in nodes[node]['childs']:
                indegrees[child]-=1
    return order


if __name__ == "__main__":
    BLOCK_CONFIG = [{'parents':[],
                     'childs':[1,2],
                     'in_channels':None,
                     'out_channels':16,
                     'kernel_size':8,
                     'stride':4},
                    {'parents':[0],
                     'childs':[],
                     'in_channels':16,
                     'out_channels':16,
                     'kernel_size':4,
                     'stride':2},
                    {'parents':[0],
                     'childs':[1],
                     'in_channels':16,
                     'out_channels':16,
                     'kernel_size':5,
                     'stride':2}]
    print(tp_order(BLOCK_CONFIG))
    