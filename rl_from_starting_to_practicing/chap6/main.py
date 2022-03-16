#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8

import gym
from puckworld import PuckWorldEnv
from agents import DQNAgent
from utils import learning_curve

env = PuckWorldEnv()
agent = DQNAgent(env, DDQN=True)

data = agent.learning(gamma=0.99,
                      epsilon=1,
                      decaying_epsilon=True,
                      alpha=1e-3,
                      max_episode_num=100,
                      display=False)

learning_curve(data, 2, 1, title="DQNAgent on PuckWorld",
               x_name="episodes", y_name="rewards")

data = agent.learning(gamma=0.99,
                      epsilon=1e-5,
                      decaying_epsilon=False,
                      alpha=1e-5,
                      max_episode_num=20,
                      display=True)