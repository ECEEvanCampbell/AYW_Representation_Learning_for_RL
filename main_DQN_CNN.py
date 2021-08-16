
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
    num_frames = 3
    update_frequency=1000

    log_dir = "/tmp/gym/"

    os.makedirs(log_dir, exist_ok=True)
    env = gym.make('LunarLander-v2')
    env.seed(0)
    img = env.render(mode='rgb_array')
    img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
    img = np.sum(img, axis=2)
    print('State shape: ', (num_frames, img.shape) )
    print('Number of actions: ', env.action_space.n)

    # 3 samples points
    

    modelname = "CNN"
    agent = Agent(modelname,gamma = 0.999, epsilon=1.0, batch_size = 64, n_actions=4,
            eps_end = 0.1, input_dims=(num_frames, *img.shape), lr=0.00001,max_mem_size=10000)
    scores, eps_history = [], []
    n_games = 2000
    frames = 0
    for i in range(n_games):
        score = 0
        done = False
        observation_ = np.zeros((num_frames,img.shape[0], img.shape[1]), dtype=np.float32)
        _ = env.reset()
        img = env.render(mode='rgb_array')
        img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
        img = np.sum(img, axis=2, dtype=np.float32)
        observation = observation_
        while not done:
            frames += 1
            action = agent.choose_action(observation)
            # We now assume the kinematics of the lander are unobservable
            _, reward, done, info = env.step(action)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img, (img.shape[0]//5, img.shape[1]//5))
            img = np.sum(img, axis=2, dtype=np.float32)
            observation_ = np.concatenate((np.expand_dims(img,0), observation[1:,:,:]))
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

