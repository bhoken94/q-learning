import numpy as np

class Agent:
    def __init__(self, lr, gamma, n_actions, n_states, eps_start, eps_end, eps_dec):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = eps_start
        self.eps_min = eps_end
        self.eps_dec = eps_dec

        # Init Q_Table
        self.Q_table = {}

        self.Q_init()

    def Q_init(self):
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.Q_table[(state, action)] = 0.0

    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            # Creo lista con le azioni possibili
            actions = [i for i in range(self.n_actions)]
            # Ne scelgo una a caso
            action = np.random.choice(actions)
        else:
            # Creo un array con
            actions = np.array([self.Q_table[(state, a)] for a in range(self.n_actions)])
            # Prendo l'indice del max action nella lista
            action = np.argmax(actions)
        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        actions = np.array([self.Q_table[(state, a)] for a in range(self.n_actions)])
        a_max = np.argmax(actions)

        # Q(s,a) = Q(s_t, a_t) + lr * (reward + gamma * max_a Q(s',a') - Q(s_t, a_t))
        self.Q_table[(state, action)] += self.lr * (reward + self.gamma * self.Q_table[(state_, a_max)] - self.Q_table[(state, action)])
        self.decrement_epsilon()
