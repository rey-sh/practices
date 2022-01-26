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

def draw_policy(policy, A, Q, epsilon, special_ace_exists=False):
    def value_of(a):
        return 0 if a == A[0] else 1

    rows, cols = 11, 10
    Z = np.zeros((rows, cols))
    dealer_first_card = np.arange(1, 12)
    play_points = np.arange(12, 22)
    for i in range(11, 22):
        for j in range(1, 11):
            s = j, i, special_ace_exists
            s = str_key(s)
            a = policy(A, s, Q, epsilon)
            Z[i-11,j-1] = value_of(a)
    
    plt.imshow(Z, cmap=plt.cm.cool, interpolation=None, origin="lower",
               extent=[0.5,11.5,10.5,21.5])
    plt.show()


def greedy_pi(A, s, Q, a):
    '''
        Give \pi(a | s) for each a.
        The case that some actions might have same value is considered.
    '''
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0/n if a in a_max_q else 0.0

def epsilon_greedy_pi(A, s, Q, a, epsilon=0.1):
    m = len(A)
    greedy_p = greedy_pi(A, s, Q, a)
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0/greedy_p)
    return (1 - epsilon) * greedy_p + epsilon/m

def epsilon_greedy_policy(A, s, Q, epsilon):
    action_prob = []
    action_num  = len(A)
    for i in range(action_num):
        action_prob.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    
    ## randomly selection according to the probability of each eaction ##
    rand_value = random.random() ## [0, 1]
    for i in range(action_num):
        rand_value -= action_prob[i]
        if rand_value < 0:
            return A[i]