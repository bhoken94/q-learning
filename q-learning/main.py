from q_learning_agent import Agent
import gym
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    agent = Agent(lr=0.001, n_actions=env.action_space.n, n_states=env.observation_space.n, eps_end=0.01, eps_dec=0.9999995, eps_start=1.0, gamma=0.9)
    scores, win_pct_list = [], []
    n_games = 5000

    for episode in range(n_games):
        done = False
        observation = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.learn(observation, action, reward, observation_)
            score += reward
            observation = observation_
        scores.append(score)
        # Ogni 100 episodi, calcolo media punteggio
        if episode % 100 == 0:
            win_perc = np.mean(scores[-100:]) # il : se messo dopo, parte dalla fine, se davanti parte dall'inizio. Il segno - comprende gli elementi
            win_pct_list.append(win_perc)
            if episode % 1000 == 0:
                # stampo info ogni 1000 episodi
                print("=> Episode {} win % {} epsilon {}".format(episode, win_perc, agent.epsilon))
    plt.plot(win_pct_list)
    plt.show()