import gym
import numpy as np

import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.wrappers import Monitor
import os
import cv2




class RandomAgent():
  def __init__(self, n_actions):
    self.action_space = [i for i in range(n_actions)]

  def choose_action(self):
    action = np.random.choice(self.action_space)
    return action



if __name__ == "__main__":
  
  env = gym.make('LunarLander-v2')
  rnd_agent = RandomAgent(n_actions=env.action_space.n)
  ideal_shape = (120,120)
  images_to_collect = 5000
  num_frames = 4 
  img_dataset_loc = 'dataset/'
  os.makedirs(img_dataset_loc,exist_ok=True)

  img = env.render(mode='rgb_array')
  img = cv2.resize(img,  ideal_shape)
  img = np.sum(img, axis=2, dtype=np.float32)

  dataset = np.zeros((images_to_collect, num_frames, *ideal_shape))
  counter = 0
  
  while counter < images_to_collect-1:
      initial_frames = 0
      done = False
      _ = env.reset()
      observation = np.zeros((num_frames,img.shape[0], img.shape[1]), dtype=np.float32)
      while not done:
        
        action = rnd_agent.choose_action()
        _, _, done, _ = env.step(action)

        img = env.render(mode='rgb_array')
        img = cv2.resize(img, ideal_shape)
        img = np.sum(img, axis=2, dtype=np.float32)/765
        observation = np.concatenate((np.expand_dims(img,0), observation[:-1,:,:])) # New frame + 2 newest frames
        
        if counter >= images_to_collect-1:
          break
        if np.random.rand(1) < 0.1 and initial_frames > num_frames:
          counter += 1
          dataset[counter,:,:,:] = observation

        initial_frames +=1

  np.save(img_dataset_loc + "dataset.npy", dataset)
  print(str(images_to_collect), " images have been collected.")

