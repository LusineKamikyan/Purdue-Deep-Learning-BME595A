#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:52:38 2018

@author: sunshinechick16
"""
import numpy as np

import matplotlib.pyplot as plt
# Here 1 represents the agent, .75 represents a dot to be eaten,
# .25 represents the ghost, .5 represents a free space,
# and 0 represents the wall that can't be passed through.

class MazeArray(object):
    def __init__(self, maze, flags, agent=(0,0)):
        self._maze = maze
        self._flags = set(flags)
        nrows, ncols = self._maze.shape
        self.maze = self._maze
        self.flags = self._flags
        self.agent = agent
        self.prev_state = agent
        
    def reset(self, agent=(0,0)):
        self.agent = agent
        self.maze = np.copy(self._maze)
        self.flags = set(self._flags)
        nrows, ncols = self.maze.shape
        row, col = agent
        self.state = ((row,col), 'start')
        self.reward ={
                'blocked':-4.0/(6**0.5),
                'flag': 1.0/32.0,
                'ghost': -8.0/(6**0.5),
                'empty': -1.0/62.0
                }
        # 0 up 
        # 1 down
        # 2 left
        # 3 right
        
    def valid_actions(self):
        (row, col) = self.agent
        actions = [0, 1, 2, 3]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(0)
        elif row == nrows-1:
            actions.remove(1)

        if col == 0:
            actions.remove(2)
        elif col == ncols-1:
            actions.remove(3)

        if row>0 and self.maze[row-1,col] == 0.0:
            actions.remove(0)
        if row<nrows-1 and self.maze[row+1,col] == 0.0:
            actions.remove(1)

        if col>0 and self.maze[row,col-1] == 0.0:
            actions.remove(2)
        if col<ncols-1 and self.maze[row,col+1] == 0.0:
            actions.remove(3)

        return actions
    # Takes an action and updates the numpy array.
    def update_state(self, action):
        nrows, ncols = self.maze.shape
        (nrow, ncol) = self.agent
        valid_actions = self.valid_actions()
        self.prev_state = (nrow,ncol)
        if action in valid_actions:
            self.maze[nrow,ncol] = .5
            if self.agent in self.flags:
                self.flags.remove(self.agent)
            if action == 0:    # move up
                nrow -= 1
            elif action == 1:  # move down
                nrow += 1
            elif action == 2:    # move left
                ncol -= 1
            elif action == 3:  # move right
                ncol += 1
        self.agent = (nrow,ncol)
        self.maze[nrow,ncol] = 1

def show_env(self, fname=None):
    plt.grid('on')
    n = self._maze.shape[0]
    ax = plt.gca()
    ax.set_xticks(np.arange(0.5, n, 1))
    ax.set_yticks(np.arange(0.5, n, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    canvas = np.copy(self.maze)
    
#    canvas[self.prev_state] = .5
#    canvas[self.agent] = 1
#    for r,c in self._flags:
#        canvas[r,c] = .75 #mark it as a flag
    img = plt.imshow(canvas, interpolation='none', cmap='gray')
    #if fname:
    #plt.savefig('/Users/anikamikyan/Desktop/BME595A/Project/a.png')
    return canvas

