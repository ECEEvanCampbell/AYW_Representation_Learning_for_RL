## This script just puts into perspective how hard it is to land the ship in the flags...


from gym.utils.play import play
import gym
import pygame



if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    play(env,fps=30,keys_to_action={(1073741903,):1,(1073741905,):2,(1073741904,):3})