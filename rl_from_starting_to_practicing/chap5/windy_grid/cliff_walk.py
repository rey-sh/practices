#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8

import utils
from gridworld import CliffWalk
from agents import SarsaAgent, QAgent

if __name__ == '__main__':
    env = CliffWalk()
    env.reset()
    
    q_agent = QAgent(env, capacity=10000)
    sarsa_agent = SarsaAgent(env, capacity=10000)
    
    sarsa_data = sarsa_agent.learning(display=False, max_episode_num=10000,
                epsilon=0.1, decaying_epsilon=False, show=True)
    
    print('-'*15)
    q_data = q_agent.learning(display=False, max_episode_num=10000,
                epsilon=0.1, decaying_epsilon=False, show=True)
    
    # n = 4
    # data = sarsa_data[2][n:], sarsa_data[1][n:], q_data[1][n:] 
    # utils.learning_curve(data, x_index = 0, y1_index = 1, y2_index = 2,
    #             title="compare of Q and Sarsa", x_name = "Episodes (n)", y_name = "Reward per Episode",
    #             y1_legend = "sarsa", y2_legend = "q learning")
    
    print('-'*15)
    sarsa_agent.last_episode_detail()
    
    print('-'*15)
    q_agent.last_episode_detail()
    
    utils.show_q(sarsa_agent)
    
    utils.show_q(q_agent)