import numpy as np

Q = np.zeros((3,3))

for _ in range(50):
    state = np.random.randint(0,3)
    action = np.argmax(Q[state])
    reward = np.random.randint(-5,10)
    Q[state, action] += 0.1 * reward

print("Q-table:", Q)
