#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import utils
from gym import Env
from core import Agent

class SarsaAgent(Agent):
    def __init__(self, env: Env = None, capacity=10000):
        super().__init__(env, capacity)
        self.Q = {}
    
    def policy(self, A, s, Q, epsilon):
        return utils.epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, gamma=0.9, alpha=0.1, epsilon=1e-5, display=False,
                        lambda_ = None, show=False):
        self.state = self.env.reset()
        s0 = self.state
        a0 = self.perform_policy(s0, self.Q, epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        
        while not is_done:
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, self.Q, epsilon)
            
            old_Q = utils.get_dict(self.Q, s0, a0)
            Q_next = utils.get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * Q_next
            new_Q = old_Q + alpha * (td_target - old_Q)
            utils.set_dict(self.Q, new_Q, s0, a0)
            
            s0, a0 = s1, a1
            time_in_episode += 1
        
        if show:
            print(self.experience.last_episode)
        
        return time_in_episode, total_reward


class SarsaLambdaAgent(Agent):
    def __init__(self, env: Env = None, capacity=10000):
        super().__init__(env, capacity)
        self.Q = {}
    
    def policy(self, A, s, Q, epsilon):
        return utils.epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, gamma=0.9, alpha=0.1, epsilon=1e-5, display=False,
                        lambda_ = None, show=False):
        self.state = self.env.reset()
        s0 = self.state
        a0 = self.perform_policy(s0, self.Q, epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        E = {}
        
        while not is_done:
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, self.Q, epsilon)
            
            q = utils.get_dict(self.Q, s0, a0)
            q_next = utils.get_dict(self.Q, s1, a1)
            delta = r1 + gamma * q_next - q
            
            e = utils.get_dict(E, s0, a0)
            e += 1
            utils.set_dict(E, e, s0, a0)
            
            for s in self.S:
                for a in self.A:
                    e_value = utils.get_dict(E, s, a)
                    old_Q = utils.get_dict(self.Q, s, a)
                    new_Q = old_Q + alpha * delta * e_value
                    new_e = gamma * lambda_ * e_value
                    utils.set_dict(self.Q, new_Q, s, a)
                    utils.set_dict(E, new_e, s, a)
            
            s0, a0 = s1, a1
            time_in_episode += 1
        
        if show:
            print(self.experience.last_episode)
        
        return time_in_episode, total_reward


class QAgent(Agent):
    def __init__(self, env: Env = None, capacity=10000):
        super().__init__(env, capacity)
        self.Q = {}
    
    def policy(self, A, s, Q, epsilon):
        return utils.epsilon_greedy_policy(A, s, Q, epsilon)

    def learning_method(self, gamma=0.9, alpha=0.1, epsilon=1e-5, display=False,
                        lambda_ = None, show=False):
        self.state = self.env.reset()
        s0 = self.state
        time_in_episode, total_reward = 0, 0
        is_done = False
        
        while not is_done:
            a0 = self.perform_policy(s0, self.Q, epsilon)
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = utils.greedy_policy(self.A, s1, self.Q) ## off-line policy ##
            
            old_Q = utils.get_dict(self.Q, s0, a0)
            Q_next = utils.get_dict(self.Q, s1, a1)
            td_target = r1 + gamma * Q_next
            new_Q = old_Q + alpha * (td_target - old_Q)
            utils.set_dict(self.Q, new_Q, s0, a0)
            
            s0 = s1
            time_in_episode += 1
        
        if show:
            print(self.experience.last_episode)
        
        return time_in_episode, total_reward