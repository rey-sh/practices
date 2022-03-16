#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import numpy as np
from gym import Env, spaces
from random import random,choice

from core import Agent
from approximator import NetApproximator

class DQNAgent(Agent):
    def __init__(self, env: Env=None, capacity=20000,
                 hidden_dim=32, batch_size=128,epochs=2, DDQN=False):
        super().__init__(env, capacity)
        
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        self.hidden_dim = hidden_dim
        self.behavior_Q = NetApproximator(input_dim=self.input_dim,
                                          output_dim=self.output_dim,
                                          hidden_dim=self.hidden_dim)
        self.target_Q = self.behavior_Q.clone()
        
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.DDQN = DDQN
    
    def _update_target_Q(self):
        self.target_Q = self.behavior_Q.clone()
    
    def policy(self, A, s=None, Q=None, epsilon=None):
        Qs = self.behavior_Q(s)
        rand_value = random()
        if epsilon is not None and rand_value < epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(Qs))
    
    def _learn_from_memory(self, gamma, learning_rate):
        trans_pieces = self.sample(self.batch_size)
        s0 = np.vstack([x.s0 for x in trans_pieces])
        a0 = np.array([x.a0 for x in trans_pieces])
        r1 = np.array([x.reward for x in trans_pieces])
        is_done = np.array([x.is_done for x in trans_pieces])
        s1 = np.vstack([x.s1 for x in trans_pieces])

        x_batch = s0
        y_batch = self.target_Q(s0)
        
        if self.DDQN:
            a_prime = np.argmax(self.behavior_Q(s1), axis=1).reshape(-1)
            Q_s1 = self.target_Q(s1)
            temp_Q = Q_s1[np.arange(len(Q_s1)), a_prime]
            Q_target = r1 + gamma * temp_Q * (~is_done)
        else:
            Q_target = r1 + gamma*np.max(self.target_Q(s1), axis=1)*(~is_done)
        
        y_batch[np.arange(len(x_batch)), a0] = Q_target
        
        loss = self.behavior_Q.fit(x=x_batch, y=y_batch,
                learning_rate=learning_rate, epochs=self.epochs)
        
        # mean_loss = loss.sum().data[0] / self.batch_size
        mean_loss = loss.sum().item() / self.batch_size
        
        self._update_target_Q()
        return mean_loss
    
    def learning_method(self, gamma=0.9, alpha=0.1, epsilon=1e-5,
                        display=False, lambda_=None):
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        
        time_in_episode, total_reward = 0, 0
        is_done = False
        loss = 0
        while not is_done:
            s0 = self.state
            a0 = self.perform_policy(s0, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            
            if self.total_trans > self.batch_size:
                loss += self._learn_from_memory(gamma, alpha)
            time_in_episode += 1
        
        loss /= time_in_episode
        
        if display:
            print("epsilon:{:3.2f}, loss:{:3.2f},{}".format(epsilon,loss,
                    self.experience.last_episode))
        
        return time_in_episode, total_reward