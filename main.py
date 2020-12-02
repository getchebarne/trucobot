import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from keras.models import load_model
from keras.optimizers import Adam
from logic import truco_qlearn
from exp_replay import ExperienceReplay
from models import create_mlp
from utils import get_action
from time import time

# parameters
num_cards = 40
state_cards = 9
epochs = 10000
epsilon = np.linspace(0.025, 0.99, num = epochs//2)  # 
epsilon = np.flip(epsilon)
epsilon = np.append(epsilon, 0.025*np.ones(epochs//2))
num_actions = 3  # [play_left, play_mid, play_right]
max_memory = 8192
batch_size = 64
input_size = state_cards + 2
lr = .0005

# optimizer
opt = Adam(lr = lr, decay = 1e-3/200)

# model
model = create_mlp(input_size = input_size, layers = [128, 256, 256, 128])
model.compile(optimizer = opt, loss = "mse")
model.summary()

# define environment
env = truco_qlearn(state_cards = state_cards, num_cards = num_cards)

# initialize experience replay object
exp_replay = ExperienceReplay(max_memory = max_memory, env_dim = input_size,
							  discount = .99)

# training loop
losses = []
ti = time()
for e in range(epochs):
    n_hands = 0
    winner = None
    actions = np.zeros((2)).astype(np.int)
    og_states = np.zeros((2, state_cards + 2))
    aux_state = None
    game_over = False
    while winner is None:
        n_hands += 1

        og_states[env.turn, :] = env.get_state().flatten()
        action_A = get_action(model, og_states[env.turn][np.newaxis], env.legal_moves(), epsilon[e])
        actions[env.turn] = action_A
        hand_over, new_state, rewards, winner = env.play_card(action_A)
        
        if aux_state is not None:
            exp_replay.remember(aux_state[np.newaxis], aux_action, 
                                 env.get_state(), rewards[int(not env.hand_winner)], game_over)

        og_states[env.turn, :] = env.get_state().flatten()
        action_B = get_action(model, og_states[env.turn][np.newaxis], env.legal_moves(), epsilon[e])
        actions[env.turn] = action_B
        hand_over, new_state, rewards, winner = env.play_card(action_B)

        aux_state = og_states[int(not env.hand_winner)]
        aux_action = actions[int(not env.hand_winner)]

        game_over = True if winner is not None else False
        # store experience
        exp_replay.remember(og_states[env.hand_winner][np.newaxis], actions[env.hand_winner], 
                            env.get_state(), rewards[env.hand_winner], game_over)
    
    # store losing experience
    exp_replay.remember(og_states[int(not env.hand_winner)][np.newaxis], actions[int(not env.hand_winner)], 
                        env.get_state(), rewards[int(not env.hand_winner)], game_over)

    # adapt model
    inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
    loss = model.train_on_batch(inputs, targets)
    #print("Player {} turn | Card played: {} | Score : {}.".format(env.turn, action, env.wins))
    losses.append(loss)
    print("Epoch {:03d}/{:03d} | Loss {:.3f} | Epsilon {:.3f}".format(e, epochs, 
            losses[-1], epsilon[e]))
    
    if e % 1000 == 0:
        for inp, tar in zip(inputs, targets):
            env.state2string(inp[:, np.newaxis])
            print("Target:", tar)
    
# Save trained model weights and architecture
print("Done training. Time elapsed: {:.2f} hours.".format((time() - ti)/3600))
print("Saving model...")
model.save("model")
print("Saved.")
np.savetxt("losses.txt", losses)
