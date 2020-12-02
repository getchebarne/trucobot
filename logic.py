import numpy as np
from sklearn.preprocessing import OneHotEncoder

class truco_qlearn:
    def __init__(self, state_cards, num_players = 2, num_cards = 40):
        self.num_players = num_players
        self.num_cards = num_cards
        self.state_cards = state_cards
        self.enc = OneHotEncoder()
        deck = np.array([-1])
        deck = np.append(deck, np.arange(num_cards))
        self.enc.fit(deck[np.newaxis].reshape((num_cards + 1, 1)))
        self.over = False
        self.win_count = np.zeros(2)
        self.deal()                                  #inicializar

    def deal(self):

        self.dealer = np.random.choice([0, 1])                   #dealer
        self.turn = int(not self.dealer)                         #turn
        self.deck = np.random.permutation(self.num_cards)        #barajar
        self.muestra = self.deck[6]
        hand_0 = [self.decode_card(h, self.muestra) for h in [self.deck[0], self.deck[2], self.deck[4]]]
        hand_1 = [self.decode_card(h, self.muestra) for h in [self.deck[1], self.deck[3], self.deck[5]]]
        hand_0 = np.sort(hand_0)
        hand_1 = np.sort(hand_1)
        self.hands = np.array([hand_0, hand_1]).astype(np.float)   #manos
        self.table = np.zeros(self.num_players).astype(np.int)              #mesa
        self.hand_score = np.zeros(self.num_players)
        self.played = np.zeros(4)
        self.played_indx = 0
        
        return self

    def play_card(self, card):
        # only legal moves are condsidered
        hand_over = False 
        game_over = False
        winner = None
        rewards = np.zeros((2))

        self.table[self.turn] = self.hands[self.turn, card]
        self.hands[self.turn, card] = 0
        self.turn = int(not self.turn)              #switch turn
        if (self.table != 0).all():
            rewards, winner = self.resolve()
            hand_over = True
        new_state = self.get_state()
        return hand_over, new_state, rewards, winner

    def resolve(self):
        
        rewards = np.zeros((self.num_players))
        winner = None
        scores = np.zeros(self.num_players)
        
        if self.played_indx <= 2:
            self.played[self.played_indx : self.played_indx + 2] = self.table.flatten()
            self.played_indx += 2

        if self.table[0] != self.table[1]:
            self.hand_winner = np.argmax(self.table)                     
            self.turn = self.hand_winner              #REVISAR
            self.hand_score[self.hand_winner] += .5
        else:                                        #fue parda
            self.hand_score += .5
            self.hand_winner = self.dealer

        if (self.hand_score == 1).any():            #game over
            over = True
            if (self.hand_score == 1).all():      #2 veces parda
                winner = self.dealer
            else:
                winner = np.argmax(self.hand_score)

            self.win_count[winner] += 1
            rewards[winner] = 1
            rewards[int(not winner)] = -1
            self.deal()
        self.table = np.zeros(self.num_players)    #reset table
        return rewards, winner

    def legal_moves(self):

        return np.where(self.hands[self.turn, :] != 0)[0]

    def get_state(self):

        state = np.zeros(self.state_cards + 2)
        state[:self.state_cards] = np.array(list(self.hands[self.turn]) + 
                                            [self.decode_card(self.muestra, self.muestra)] +
                                            [self.table[int(not self.turn)]] +
                                            list(np.sort(self.played)))
        
        state /= 19
        state[self.state_cards:] = np.array([self.hand_score[self.turn], self.hand_score[int(not self.turn)]])
        return state[np.newaxis]

    def state2string(self, state = "current"):

        if state is "current":
            state = self.get_state()

        state = state.flatten()
        print("Hand: {} | Muestra: {} | Table: {} | Score: {}".format(np.round(state[:3].T, 2).flatten(),
                                                                      np.round(state[3], 2),
                                                                      np.round(state[4], 2),
                                                                      state[-2:].T))
        print("Played:", np.round(state[5:-2].T, 2).flatten())

    '''
    def get_state(self):

        state = np.zeros((self.state_cards*(self.num_cards + 1) + 2))

        aux = np.array(self.hands[self.turn, :])
        aux = np.append(aux, self.muestra)
        #aux = np.append(aux, self.table[self.turn])
        aux = np.append(aux, self.table[int(not self.turn)])
        aux = np.append(aux, self.played)[:, np.newaxis]
        state[0 : self.state_cards*(self.num_cards + 1)] = self.enc.transform(aux).toarray().flatten()
        state[self.state_cards*(self.num_cards + 1)] = self.hand_score[self.turn]
        state[self.state_cards*(self.num_cards + 1) + 1] = self.hand_score[int(not self.turn)]

        return state[np.newaxis]

    def state2string(self, state = "current"):
        if state is "current":
            state = self.get_state().T

        aux = state[:-2].reshape((self.state_cards, self.num_cards + 1))
        state_num = self.enc.inverse_transform(aux).flatten()
        print("Hand: {} | Muestra: {} | Table: {} | Score: {}".format([self.card2string(h) for h in state_num[:3]],
                                                                       self.card2string(state_num[3]),
                                                                       self.card2string(state_num[4]),
                                                                       state[-2:].T))
        print("Played:", [self.card2string(p) for p in state_num[5:9]])
        #print("---------------------------------------------------------------------------------------------------------------")
    '''
    @staticmethod               
    def decode_card(card, muestra):
            
        piezas = [1, 3, 4, 7, 8]
        palo_card = card // 10
        palo_muestra = muestra // 10
        value = card % 10
        card_score = value - 2 if card != 0 else 0                           #carta "normal"
        if palo_card == palo_muestra and value in piezas:   #es pieza
            if value == 1:
                card_score = 19
            elif value not in [7, 8]:
                card_score = 21 - value
            else:                                         #es 10 o 11 
                card_score = value + 8
        elif (value == 0 and (palo_card == 1 or palo_card == 2)): #es 7 de mata
            card_score = 15 - palo_card
        elif (value == 6 and (palo_card == 0 or palo_card == 1)): #es 1 de mata
            card_score = 11 + palo_card
        elif value == 0 or value == 1 or value == 2: #es 1, 2 o 3
            card_score = 8 + value

        return card_score

    @staticmethod
    def card2string(card):
        val = (card % 10) + 1
        palo = card // 10
        if card != -1:  
            decode_palo = {-1 : "nada.", 0 : "oro", 1 : "espada", 2 : "basto", 3 : "copa"}  
            if val > 7: val += 2 
            return str(int(val)) + "_" + decode_palo[palo]
        else:
            return "na"