import gym
from gym import wrappers
from utils import Agent
from utils import plot_learning_curve
import numpy as np



if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    agent = Agent(gamma = 0.99, epsilon=1.0, batch_size = 64, n_actions=4,
            eps_end = 0.01, input_dims=[8], lr=0.001)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('episode ', i, 'score %.2f' % score, 
                'average score %.2f' % avg_score,
                'epsilon %.2f' %agent.epsilon)
    
    x = [i+1 for i in range(n_games)]
    filename = 'lunar_lander_2020.png'
    plot_learning_curve(x, scores, eps_history, filename)

    # Render the final Agent's behaviour
    env = wrappers.Monitor(env, "./gym-results",force=True)
    observation = env.reset()
    for _ in range(1000):
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        if done: break
    env.close()