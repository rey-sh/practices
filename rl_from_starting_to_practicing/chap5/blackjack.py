#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import utils
import random
from tqdm import tqdm
from queue import Queue

class Gamer():
    '''
        A basic gamer class can:
            - hand cards and calculate points
            - receive cards, or discharge cards when a game overs
    '''
    def __init__(self, name="", display=False):
        self.name = name
        self.display = display
        self.role = ""
        self.action_set = ["Bid", "Stop"]
        
        self.cards = []
    
    def __str__(self):
        return self.name
    
    def _info(self, message):
        if self.display:
            print(message, end="")

    def _card_value(self, card):
        '''
            Get the value of a card: 2-9 go without saying; A is 1, JQK are 10
        '''
        if card == 'A':
            v = 1
        elif card in ['J', 'Q', 'K']:
            v = 10
        else:
            v = int(card)
        
        return v
    
    def calc_points(self):
        '''
            Calculate the points of the cards in hand.
            Return:
                - total_point:       the maximal points of the current cards.
                - special_ace_exists: whethere there is a Ace regarded as 11
        '''
        special_ace_num = 0
        total_point = 0
        if self.cards is None:
            return 0, False
        
        for c in self.cards:
            v = self._card_value(c)
            ## ace will be preferentially regarded as 11 (special ace)
            if v == 1:
                special_ace_num += 1
                v = 11
            total_point += v
        
        ## check if any ace can be reduced to 1 (normal ace) if point > 21 ##
        while total_point > 21 and special_ace_num > 0:
            total_point -= 10
            special_ace_num -= 1
        
        special_ace_exists = (special_ace_num != 0)
        
        return total_point, special_ace_exists
    
    def receive_cards(self, cards = []):
        '''
            Receive new cards.
        '''
        cards = list(cards)
        self.cards.extend(cards)
    
    def discharge_cards(self):
        '''
            Discharged cards when game overs.
        '''
        self.cards.clear() 
    
    def cards_info(self):
        '''
            Show the cards in hand.
        '''
        self._info("{}.{}'s cards are:{}\n".format(self.role, self.name, self.cards))


class Dealer(Gamer):
    def __init__(self, name="", display=False):
        super().__init__(name, display)
        self.role = "Dealer"
    
    def show_first_card(self):
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._card_value(self.cards[0])
    
    def policy(self):
        points, _ = self.calc_points()
        if points < 17:
            action = self.action_set[0]
        else:
            action = self.action_set[1]
        return action


class Player(Gamer):
    def __init__(self, name="", display=False):
        super().__init__(name, display)
        self.role = "Player"
        self.learning_method = None
    
    def get_state(self, dealer):
        dealer_first_card = dealer.show_first_card()
        points, special_ace_exists = self.calc_points()
        return (dealer_first_card, points, special_ace_exists)
    
    def get_state_name(self, dealer):
        return utils.str_key(self.get_state(dealer))
    
    def policy(self, dealer=None):
        points, _ = self.calc_points()
        if points < 20:
            action = self.action_set[0]
        else:
            action = self.action_set[1]
        return action

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
            player_action = player.policy(dealer)
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
        self.recycle_cards(player, dealer)
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
            ## drew is regarded as 1/2 win ##
            print("Winning percentage: {:.2f}%".format(
                ((results[2] + 0.5*results[1])/num) * 100))
    
    def _info(self, message):
        if self.display:
            print(message, end="")