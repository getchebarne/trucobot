import numpy as np

def get_action(model, state, legal_moves, epsilon):

    if np.random.rand() < epsilon:
        a = int(np.random.choice(legal_moves))
    else:
        q = model.predict(state)[0]
        aux = np.argmax(q[legal_moves])
        a = legal_moves[aux]

    return a
