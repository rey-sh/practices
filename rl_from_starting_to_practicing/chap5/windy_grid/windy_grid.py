#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import gym
from gym import Env
import matplotlib.pyplot as plt

from core import Agent
from gridworld import WindyGridWorld
from agents import SarsaAgent, SarsaLambdaAgent, QAgent

if __name__ == '__main__':
    env = WindyGridWorld()
    env.reset()
    
    # agent = Agent(env, capacity=10000)
    # data = agent.learning(max_episode_num=180, display=False)
    # times = [data[0][0]]
    # times.extend([data[0][i+1] - data[0][i] for i in range(0,180-1)])
    # plt.plot(range(180), times)
    # plt.show()
    
    agent = SarsaAgent(env, capacity=100000)
    statistics = agent.learning(lambda_=0.8, gamma=1.0, epsilon=0.2,\
        decaying_epsilon=True, alpha=0.5, max_episode_num=800, display=False,\
        show=True)
    print("-"*120)
    
    agent = SarsaLambdaAgent(env, capacity=100000)
    statistics = agent.learning(lambda_=0.8, gamma=1.0, epsilon=0.2,\
        decaying_epsilon=True, alpha=0.5, max_episode_num=800, display=False,\
        show=True)
    print("-"*120)
    
    agent = QAgent(env, capacity=100000)
    statistics = agent.learning(lambda_=0.8, gamma=1.0, epsilon=0.2,\
        decaying_epsilon=True, alpha=0.5, max_episode_num=800, display=False,\
        show=True)