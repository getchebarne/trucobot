import numpy as np 
import json 
from keras.models import load_model
from logic import truco_qlearn

num_cards = 40
state_cards = 9
model = load_model("model")
env = truco_qlearn(state_cards = state_cards, num_cards = num_cards)
n_games = 100000
winc = np.zeros((2))
for n in range(n_games):
    if n % 5000 == 0:
        print("{}/{} games played.".format(n + 1, n_games))
    #print("--------------------------- START ------------------------------------")
    env.deal()
    game_over = False
    while not game_over:
        if env.turn == 0:
            action = np.random.choice(env.legal_moves())
        else:
            q = model.predict(env.get_state())[0]
            aux = np.argmax(q[env.legal_moves()])
            action = env.legal_moves()[aux]
            
            #env.state2string()
            #print("Q:", np.round(q, 2))
            #print("action:", action)
            
        hand_over, new_state, rewards, winner = env.play_card(action)
        if winner is not None: game_over = True
    winc[winner] += 1
    #print("----------------------------- END -------------------------------------")
print("{:.2f} % winrate.".format(winc[1]*100/n_games))