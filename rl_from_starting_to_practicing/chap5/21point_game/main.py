#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import math
import utils
from blackjack import Player, Dealer, Arena

class MC_Player(Player):
    '''
        Player with MC policy control.
    '''
    def __init__(self, name="", display=False):
        super().__init__(name, display)
        self.Q = {} ## Q function
        self.Nsa = {} ## (s, a) pair counter
        self.total_learning_times = 0
        self.learning_method = self.learn_Q
    
    def learn_Q(self, episode, reward):
        '''
            Study with a collected episode.
        '''
        for s, a in episode:
            nsa = utils.get_dict(self.Nsa, s, a)
            utils.set_dict(self.Nsa, nsa+1, s, a)
            q = utils.get_dict(self.Q, s, a)
            utils.set_dict(self.Q, q + (reward-q)/(nsa+1), s, a)
        self.total_learning_times += 1
        
    def reset_momery(self):
        '''
            Clear the experience learned.
        '''
        self.Q.clear()
        self.Nsa.clear()
        self.total_learning_times = 0
    
    def policy(self, dealer, epsilon=None):
        '''
            ε-greedy learning policy.
        '''
        player_points, _ = self.calc_points()
        if player_points >= 21:
            return self.action_set[1]
        if player_points < 12:
            return self.action_set[0]
        else:
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0/(1 + 4*math.log10(1+self.total_learning_times))
            return utils.epsilon_greedy_policy(self.action_set, s, self.Q, epsilon)

if __name__ == '__main__':
    display = False
    
    player = MC_Player(display=display, name="Elio")
    dealer = Dealer(display=display, name="God")
    arena = Arena(display=display)
    
    arena.repeat_game(dealer, player, num=200000)
    
    utils.draw_policy(utils.epsilon_greedy_policy, player.action_set, player.Q,
                      epsilon=1e-10, special_ace_exists=False)
    
    utils.draw_policy(utils.epsilon_greedy_policy, player.action_set, player.Q,
                      epsilon=1e-10, special_ace_exists=True)