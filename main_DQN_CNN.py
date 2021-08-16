
import gym
from gym import wrappers
from utils import Agent
from utils import plot_learning_curve
import numpy as np
import cv2
import torch
import os


if __name__ == '__main__':
    RENDER = False

    log_dir = "/tmp/gym/"

    os.makedirs(log_dir, exist_ok=True)
    env = gym.make('LunarLander-v2')
    env.seed(0)
    img = env.render(mode='rgb_array')
    img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
    img = np.sum(img, axis=2)
    print('State shape: ', img.shape)
    print('Number of actions: ', env.action_space.n)

    observation_ = np.float32(img)


    modelname = "CNN"
    agent = Agent(modelname,gamma = 0.999, epsilon=1.0, batch_size = 64, n_actions=4,
            eps_end = 0.1, input_dims=img.shape, lr=0.00005,max_mem_size=10000)
    scores, eps_history = [], []
    n_games = 500

    for i in range(n_games):
        score = 0
        done = False
        _ = env.reset()
        img = env.render(mode='rgb_array')
        img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
        img = np.sum(img, axis=2, dtype=np.float32)
        observation = img
        while not done:
            action = agent.choose_action(observation)
            # We now assume the kinematics of the lander are unobservable
            _, reward, done, info = env.step(action)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
            img = np.sum(img, axis=2, dtype=np.float32)
            observation_ = img
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

