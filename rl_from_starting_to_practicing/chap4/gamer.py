#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# Python version: 3.8
import utils

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
    
    def policy(self):
        points, _ = self.calc_points()
        if points < 20:
            action = self.action_set[0]
        else:
            action = self.action_set[1]
        return action