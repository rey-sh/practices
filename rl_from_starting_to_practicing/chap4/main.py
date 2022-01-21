#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import env
import utils
import gamer

def policy_evaluate(episodes, V, Ns):
    for episode, r in episodes:
        for s, a in episode:
            ns = utils.get_dict(Ns, s)
            v = utils.get_dict(V, s)
            utils.set_dict(Ns, ns+1, s)
            ## using r instead of G_t is because this is a one-step game
            utils.set_dict(V, v+(r-v)/(ns+1), s)

if __name__ == '__main__':
    display = False
    
    player = gamer.Player(display=display, name="Elio")
    dealer = gamer.Dealer(display=display, name="God")
    
    arena = env.Arena(display=display)
    
    arena.repeat_game(dealer, player, num=200000)
    
    V = {} ## value function
    Ns = {} ## visiting time counter
    policy_evaluate(arena.episodes, V, Ns)