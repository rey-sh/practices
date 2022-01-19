#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8

'''
    1. Toy projects for Reinforcement Learning.
    2. Focus on POLICY ITERATION and VALUE ITERATION of MDP.
    3. Partially follow to the codes by QiangYe:
    https://github.com/qqiang00/Reinforce/tree/master/reinforce/codes_for_book/c03
'''

## enviroment ##
def env(s, a):
    '''
        Args:
            - s:    current state
            - a:    selected action
        Rerturns:
            - s_next:   the next state
            - r:        reward
            - is_end:   whether the agent has entering the terminate state
    '''
    
    assert (s in S), f"Illegal state: {s}"
    assert (a in A), f"Illegal action: {a}"
    
    ## Special cases: 1) if the agent are going to get out of boundary, it will
    ##                   be forced to stay at the same position (state);
    ##                2) if the agent has entering the terminate state, it will
    ##                   stays at there forever.
    if (s%4 == 0 and a == 'w') or (s<4 and a == 'n')  or \
        ((s+1)%4 == 0 and a == 'e') or (s>11 and a == 's') or \
        (s in [0, 15]):
            s_next = s
    
    else:
        s_next = s + a_on_s[a]
    
    r = 0 if s in [0, 15] else -1
    is_end = (s in [0, 15])
    
    return s_next, r, is_end

## state transition probability ##
def P(s, s1, a):
    s_next, _, _ = env(s, a)
    return s1 == s_next

## reward function ##
def R(s, a):
    _, r, _ = env(s, a)
    return r


class uniform_random_policy():
    def __init__(self, MDP):
        self.S, self.A, self.R, self.P, self.gamma = MDP
        ## initial (state) value function ##
        self.V = [0 for _ in range(len(self.S))]
        self.update_policy()
    
    def update_policy(self):
        self.policy = {}
        for s in self.S:
            p = {a: 1.0/len(self.A) for a in self.A}
            self.policy[s] = p
    
    def policy_evaluate(self, N = 1):
        for n in range(N):
            self.update_value()
    
    def update_value(self):
        '''
            Update the value function according to the Bellman Equation.
        '''
        V = self.V.copy()
        for s in self.S:
            v_new = 0
            ## get V_{k+1}(s) from V_{k}(s) ##
            for a in self.A:
                v_next_sum = 0
                p = self.policy[s][a]
                for s_next in self.S:
                    v_next_sum += self.P(s, s_next, a) * V[s_next]
                v_new += p * (R(s,a) + self.gamma * v_next_sum)
            self.V[s] = v_new
    
    def display_value(self):
        for i in range(len(self.V)):
            print('{0:>6.2f}'.format(self.V[i]), end = " ")
            if (i+1) % 4 == 0:
                print("")
        print()
    
    def display_policy(self):
        for s in self.S:
            p = ""
            print("State {:0>2}：".format(s), end=" ")
            for a in self.A:
                if self.policy[s][a] != 0:
                    p += "{} ({:^4.2f}) ".format(a, self.policy[s][a])
            print(p)
        

class greedy_policy(uniform_random_policy):
    def __init__(self, MDP):
        super().__init__(MDP)

    def update_policy(self):
        '''
            Give the greedy policy according to the current value function.
        '''
        ## calculate the action-value function q(s, a) ##
        self.policy = {}
        
        Q = {}
        for s in self.S:
            q = {}
            for a in self.A:
                v_next_sum = 0
                for s_next in self.S:
                    v_next_sum += self.P(s, s_next, a) * self.V[s_next]
                q[a] = self.R(s, a) + self.gamma * v_next_sum
            Q[s] = q
        
        ## determine the newer and better policy ##
        for s in self.S:
            p = {}
            max_q = max(Q[s].values())
            max_cnt = 0
            for a in self.A:
                if (Q[s][a] == max_q):
                    p[a] = 1.0
                    max_cnt += 1
                else:
                    p[a] = 0.0
            p = {k: v/max_cnt for k,v in p.items()}
            self.policy[s] = p
    
    def policy_iterate(self, eval_N = 1, N = 1):
        '''
            Implement policy itertaion for 'N' times.
            For each time, do policy evaluation for 'eval_N' times and then
            update the policy in a greedy way.
        '''
        for _ in range(N):
            self.policy_evaluate(eval_N)
            self.update_policy()
    
    def value_iterate(self, N = 1):
        '''
            Implement value iteration for 'N' times.
        '''
        for _ in range(N):
            V = self.V.copy()
            for s in self.S:
                v_new = -float('inf')
                for a in self.A:
                    v_next_sum = 0
                    for s_next in self.S:
                        v_next_sum += self.P(s, s_next, a) * V[s_next]
                    v_new = max(self.R(s, a)+self.gamma*v_next_sum, v_new)
                self.V[s] = v_new


if __name__ == '__main__':
    ITER_NUM = 200
    
    ## state space ##
    S = [i for i in range(16)]

    ## action space ##
    A = ['n', 'e', 's', 'w']
    a_on_s = {'n': -4, 'e': +1, 's': +4, 'w': -1} ## impact of the action on the state
    
    gamma = 1.00
    MDP = S, A, R, P, gamma
    
    UP = uniform_random_policy(MDP)
    UP.policy_evaluate(ITER_NUM)
    print("Value Function of Uniform Random Policy (after 100 evaluation):")
    UP.display_value()
    
    GP1 = greedy_policy(MDP)
    GP1.policy_iterate(1, 100)
    print("Value Function of Greedy Policy (after 100 policy iterations):")
    GP1.display_value()
    
    GP2 = greedy_policy(MDP)
    GP2.value_iterate(4)
    print("Value Function of Greedy Policy (after 4 value iterations):")
    GP2.display_value()
    GP2.update_policy()
    print("Optimal Policy using Greddy Value Iteration:")
    GP2.display_policy()