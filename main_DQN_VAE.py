
import gym
from gym import wrappers
from utils import Agent
from utils import plot_learning_curve
import numpy as np
import cv2
import torch
import os

from main_vae import ConvVAE

def init_vaeQlearner(vae, agent):
    agent.Q_eval.conv1.weight = vae.conv1.weight
    agent.Q_eval.conv1.bias   = vae.conv1.bias
    agent.Q_eval.conv2.weight = vae.conv2.weight
    agent.Q_eval.conv2.bias   = vae.conv2.bias
    agent.Q_eval.conv3.weight = vae.conv3.weight
    agent.Q_eval.conv3.bias   = vae.conv3.bias
    agent.Q_eval.fc1.weight   = vae.q_fc_mu.weight
    agent.Q_eval.fc1.bias     = vae.q_fc_mu.bias
    # Make the pretrained layers not trainable
    agent.Q_eval.conv1.eval()
    agent.Q_eval.conv2.eval()
    agent.Q_eval.conv3.eval()
    agent.Q_eval.fc1.eval()
    # Update the target network to the eval network
    agent.update_target()
    return agent



def Transform_Image(_img, _n):
    # Highlight what we care about -- lunar lander.
    _x = _img
    _x = _x * (_x < 0.8) * (_x > 0.4) # Mask anything outside 0.4 < _x < 0.8 
    #_x[50:52, 32:34] = 0 # remove flags
    #_x[50:52, 48:50] = 0
    _y = _n * _x + _img # highlight items between region, add to original image.

    return _y


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_frames = 4
    ideal_img_size = (120,120)
    # Load the pretrained VAE encoder
    modelname="VAE"

    vae_model_location = 'VAE_Checkpoint_Lunar.pt'
    vae = ConvVAE(K=18, input_dims=(num_frames, *list(ideal_img_size)), filter1=32, filter2=64)
    vae.load_state_dict(torch.load(vae_model_location))

    agent = Agent(modelname,gamma = 0.99, epsilon=1.0, batch_size = 32, n_actions=4,
            eps_end = 0.1, input_dims=(num_frames, *list(ideal_img_size)), lr=0.0001,max_mem_size=3000)

    # Initialize Q_eval as VAE encoder
    agent = init_vaeQlearner(vae,agent)
    agent.Q_eval.to(device)

    RENDER = False
    
    update_frequency=1000
    
    log_dir = "/tmp/gym/"

    os.makedirs(log_dir, exist_ok=True)
    env = gym.make('LunarLander-v2')
    env.seed(0)
    img = env.render(mode='rgb_array')
    img = cv2.resize(img, ideal_img_size) 
    img = np.sum(img, axis=2)
    print('State shape: ', (num_frames, img.shape) )
    print('Number of actions: ', env.action_space.n)

    # 3 samples points
    
    scores, eps_history = [], []
    n_games = 1000
    frames = 0
    for i in range(n_games):
        score = 0
        done = False
        observation_ = np.zeros((num_frames,img.shape[0], img.shape[1]), dtype=np.float32)
        _ = env.reset()
        img = env.render(mode='rgb_array')
        img = cv2.resize(img,  ideal_img_size)
        img = np.sum(img, axis=2, dtype=np.float32) / 765
        img = Transform_Image(img, 3)
        observation = observation_
        while not done:
            frames += 1
            action = agent.choose_action(observation)
            # We now assume the kinematics of the lander are unobservable
            _, reward, done, info = env.step(action)
            img = env.render(mode='rgb_array')
            img = cv2.resize(img,  ideal_img_size)
            img = np.sum(img, axis=2, dtype=np.float32)/765
            img = Transform_Image(img, 3)
            observation_ = np.concatenate((np.expand_dims(img,0), observation[:-1,:,:])) # New frame + 2 newest frames
            score += reward
            agent.store_transitions(observation, action, reward, observation_, done)
            if frames % num_frames:
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

