import numpy as np

class ExperienceReplay:
    def __init__(self, env_dim, max_memory=100, discount=.99):
        self.max_memory = max_memory
        self.env_dim = env_dim
        self.states = np.zeros((max_memory, env_dim))
        self.states_tp1 = self.states.copy()
        self.rewards = np.zeros((max_memory,))
        self.actions = np.zeros((max_memory,))
        self.game_overs = np.zeros((max_memory,))
        self.discount = discount
        self.indx = 0
        self.looped = False

    def remember(self, state, action, state_tp1, reward, game_over):
        self.states[self.indx] = state.flatten()
        self.actions[self.indx] = action
        self.states_tp1[self.indx] = state_tp1.flatten()
        self.rewards[self.indx] = reward
        self.game_overs[self.indx] = game_over
        self.indx += 1
        if self.indx >= (self.max_memory - 1):
            self.indx = 0
            self.looped = True

    def get_batch(self, model, batch_size=10):
        num_actions = model.output_shape[-1]
        len_memory = self.indx if not self.looped else self.max_memory
        inputs = np.zeros((min(len_memory, batch_size), self.env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        #self.print_mem()
        #print(len_memory)
        for i, idx in enumerate(np.random.randint(0, len_memory,
                                                  size=inputs.shape[0])):
            state_t = self.states[idx][np.newaxis]
            action_t = int(self.actions[idx])
            state_tp1 = self.states_tp1[idx][np.newaxis]
            reward_t = self.rewards[idx]
            game_over = int(self.game_overs[idx])
            '''
            print("!St:", asd.state2string(state_t.reshape((248, 1))))
            print("!a:", action_t)
            print("!St+1:", asd.state2string(state_tp1.reshape((248, 1))))
            print("!r:", reward_t)
            print("!game over:", game_over)
            print("idx", idx)
            print("----------------------------------------------------------------------------------")
            '''
            inputs[i:i+1] = state_t
            # There should be no target values for actions not taken.
            # Thou shalt not correct actions not taken #deep
            targets[i] = model.predict(state_t)[0]
            if game_over:  # if game_over is True
                targets[i, action_t] = reward_t
            else:
                # reward_t + gamma * max_a' Q(s', a')
                Q_sa = np.max(model.predict(state_tp1)[0])
                targets[i, action_t] = reward_t + self.discount * Q_sa
        return inputs, targets