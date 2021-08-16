import gym
from gym import wrappers
from utils import Agent
from utils import plot_learning_curve
import numpy as np
import os
import torch


if __name__ == '__main__':
    RENDER = False
    update_frequency = 1000

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    modelname = "MLP"
    agent = Agent(modelname,gamma = 0.99, epsilon=1.0, batch_size = 64, n_actions=4,
        eps_end = 0.1, input_dims=[8], lr=0.0005)
    scores, eps_history = [], []
    n_games = 500
    frames = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            frames += 1

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            agent.learn()
            if frames % update_frequency:
                agent.update_target()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-10:])

        print('episode ', i, 'score %.2f' % score, 
                'average score %.2f' % avg_score,
                'epsilon %.2f' %agent.epsilon)
    
    x = [i+1 for i in range(n_games)]
    filename = modelname + '_lunar_lander_2020.png'
    plot_learning_curve(x, scores, eps_history, filename)
    os.makedirs('/models/',exist_ok=True)
    torch.save(agent.Q_eval.state_dict(), f'/models/{modelname}Brain_{n_games}.dqn')
    np.save(f"{modelname}Brain_{n_games}_results.npy", scores)

    if RENDER:
        # Render the final Agent's behaviour
        env = wrappers.Monitor(env, "./gym-results",force=True)
        observation = env.reset()
        for _ in range(1000):
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if done: break
        env.close()