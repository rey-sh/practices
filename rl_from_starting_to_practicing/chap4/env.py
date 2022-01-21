#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import random
from tqdm import tqdm
from queue import Queue

class Arena():
    '''
        A class to manage all the game.
    '''
    def __init__(self, display=None):
         self.cards = ['A','2','3','4','5','6','7','8','9','J','Q','K'] * 4
         self.action_set = ["Bid", "Stop"]
         self.card_pool = Queue(maxsize=52)
         self.discarded_cards = []
         
         self.display = display
         self.episodes = []
         self.load_cards(self.cards)
    
    def load_cards(self, cards):
        '''
            Shuffle the cards and load them into the card pool.
        '''
        random.shuffle(cards)
        for c in cards:
            self.card_pool.put(c)
        cards.clear()
    
    def give_reward(self, dealer, player):
        '''
            Determine the reward to the player.
        '''
        dealer_points, _ = dealer.calc_points()
        player_points, special_ace_exists = player.calc_points()
        
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        
        return reward, player_points, dealer_points, special_ace_exists
    
    def serve_card(self, player, n=1):
        '''
            Deal 'n' cards to the 'player'.
            Once there are insufficient cards in the pool, Arena will load cards
        '''
        cards = []
        for _ in range(n):
            if self.card_pool.empty(): ## no card in the pool ##
                self._info("No card in the pool.")
                random.shuffle(self.discarded_cards)
                self._info("{} discarded cards are shuffed and loaded again.".format(
                    len(self.discarded_cards)))
                ## the players are not allowed to have too many cards in hand
                ## which means when the card pool is empty, there must have
                ## been enough cards be discarded.
                assert(len(self.discarded_cards) > 20)
                self.load_cards(self.discarded_cards)
            cards.append(self.card_pool.get())
        self._info("Deal {} cards ({}) to {}.{}\n".format(
            n, cards, player.role, player.name))
        player.receive_cards(cards)
        player.cards_info()
        self._info("\n")
        
    
    def recycle_cards(self, *players):
        if len(players) == 0:
            return
        for p in players:
            for c in p.cards:
                self.discarded_cards.append(c)
            p.discharge_cards()
    
    def play_game(self, dealer, player):
        '''
            Play one game.
        '''
        self._info("======== New game =========\n")
        self.serve_card(player, n=2)
        self.serve_card(dealer, n=2)
        episode = []
        ## game begins ##
        while True:
            player_action = player.policy()
            self._info("{}.{} decides to: {}\n".format(
                player.role, player.name, player_action))
            episode.append((player.get_state_name(dealer), player_action))
            if player_action == self.action_set[0]:
                self.serve_card(player)
            else:
                break
        
        ## the player's points gets overflowed ##
        reward, player_points, _, _ = self.give_reward(dealer, player)
        if player_points > 21:
            self._info("The player got {} points and lost, reward: {}.\n".format(
                player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward))
            self._info("======== Game over ========\n")
            return episode, reward
        
        ## does not get overflowed ##
        self._info("\n") 
        while True:
            dealer_action = dealer.policy()
            self._info("{}.{} decides to: {}\n".format(
                dealer.role, dealer.name, dealer_action))
            if dealer_action == self.action_set[0]:
                self.serve_card(dealer)
            else:
                break
        
        ## both stop bidding ##
        self._info("\n")
        self._info("BIDDING STOPPED.")
        self._info("\n\n")
        
        reward, player_points, dealer_points, _ = self.give_reward(dealer,player)
        player.cards_info()
        dealer.cards_info()
        self._info("Player: {}, Dealer: {}\n".format(player_points, dealer_points))
        if reward == +1:
            self._info("The player won!\n")
        elif reward == -1:
            self._info("The dealer won!\n")
        else:
            self._info("This game is a draw.\n")
        self.episodes.append((episode,reward))
        self._info("======== Game over ========\n")
        
        return episode, reward
    
    def repeat_game(self, dealer, player, num=2, show_statistic=True):
        '''
            Play game for 'num' rounds.
        '''
         ## the number of rounds that the player wins/draws/losses
        results = [0, 0, 0]
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        
        if show_statistic:
            print("Total {} rounds: The player won/drew/lost for {}/{}/{} rounds".format(
                num, results[2], results[1], results[0]))
            print("Win rate: {:.2f}".format(results[2]/num))
    
    def _info(self, message):
        if self.display:
            print(message, end="")