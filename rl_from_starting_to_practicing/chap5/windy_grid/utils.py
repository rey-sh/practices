#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import random
import numpy as np
import matplotlib.pyplot as plt

def str_key(*args):
    '''将参数用"_"连接起来作为字典的键，需注意参数本身可能会是tuple或者list型，
    比如类似((a,b,c),d)的形式。
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value

def get_dict(target_dict, *args):
    return target_dict.get(str_key(*args),0)

def sample(A):
    return random.choice(A)

def greedy_policy(A, s, Q, epsilon=None):
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)

def epsilon_greedy_policy(A, s, Q, epsilon=0.05):
    rand_value = random.random()
    if rand_value < epsilon:
        return sample(A)
    else:
        return greedy_policy(A, s, Q)

def learning_curve(data, x_index = 0, y1_index = 1, y2_index = None, title = "", 
                   x_name = "", y_name = "",
                   y1_legend = "", y2_legend=""):
    '''根据统计数据绘制学习曲线，
    Args:
        statistics: 数据元组，每一个元素是一个列表，各列表长度一致 ([], [], [])
        x_index: x轴使用的数据list在元组tuple中索引值
        y_index: y轴使用的数据list在元组tuple中的索引值
        title:图标的标题
        x_name: x轴的名称
        y_name: y轴的名称
        y1_legend: y1图例
        y2_legend: y2图例
    Return:
        None 绘制曲线图
    '''
    fig, ax = plt.subplots()
    x = data[x_index]
    y1 = data[y1_index]
    ax.plot(x, y1, label = y1_legend)
    if y2_index is not None:
        ax.plot(x, data[y2_index], label = y2_legend)
    ax.grid(True, linestyle='-.')
    ax.tick_params(labelcolor='black', labelsize='medium', width=1)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    ax.legend()
    plt.show()

def xy2state(x, y):
    return str(y*12+x)

def str_key(s, a):
    '''根据横坐标，纵坐标和行为生成键
    '''
    return str(s)+"_"+str(a)
    
def print_q(agent):
    '''打印输出agent的价值
    '''
    for y in range(4):
        for x in range(12):
            for a in range(4):
                key = str_key(xy2state(x,y),a)
                print("{}_{}_{}:{}".format(x,y,a,agent.Q.get(key,0)))
                
def show_q(agent):
    '''绘制agent学习得到的Q值，以图片的形式，每一个位置用3*3的小方格表示，
    中间小方格表示该状态的价值，左右上下四个小方格分别表示相应行为的价值，
    四个角上的数据暂时没有意义。
    '''
    V = np.zeros((4*3,12*3))
    for y in range(4):
        for x in range(12):
            max_qsa = -float('inf')
            for a in range(4): # 0-3 分别为 左 右 上 下
                key = str_key(xy2state(x,y),a)
                qsa = agent.Q.get(key,0)
                if a == 0: V[3*y+1, 3*x+1-1] = qsa
                if a == 1: V[3*y+1, 3*x+1+1] = qsa
                if a == 2: V[3*y+1+1, 3*x+1] = qsa
                if a == 3: V[3*y+1-1, 3*x+1] = qsa
                if qsa > max_qsa: max_qsa = qsa
            V[3*y+1, 3*x+1] = max_qsa
    plt.imshow(V, cmap=plt.cm.gray, interpolation=None, origin="lower", extent=[0, 12, 0, 3])
    plt.show()