import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class MLPQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(MLPQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.n_actions  = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss      = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions



class CNNQNetwork(nn.Module):
    def __init__(self, lr, input_dims, filter1, filter2, fc1n, n_actions):
        super(CNNQNetwork, self).__init__()

        self.input_dims = input_dims
        self.filter1   = filter1
        self.filter2   = filter2
        self.n_actions  = n_actions

        self.conv1   = nn.Conv2d(1, self.filter1,7, stride=2)
        self.conv2   = nn.Conv2d(self.filter1, self.filter2,5, stride=1)
        self.maxpool = nn.MaxPool2d(4)
        if 'neurons' not in locals():
          neurons = self.getNeuronNum(torch.zeros(1,1,*input_dims))
        self.fc1     = nn.Linear(neurons, fc1n)
        self.fc2     = nn.Linear(fc1n, self.n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss      = nn.MSELoss()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        
        x = self.maxpool(F.relu(self.conv1(state)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        actions = self.fc2(x)
        
        return actions
    def getNeuronNum(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return x.numel()


class Agent():
    def __init__(self, modelname, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end=0.01, eps_dec=1e-4):
        self.modelname = modelname
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        if modelname == "MLP":
          self.Q_eval = MLPQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, 
                                    fc1_dims=256, fc2_dims=256)
        elif modelname == "CNN":
          self.Q_eval = CNNQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                    filter1=32, filter2=32, fc1n=32)
        self.state_memory     = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        self.action_memory    = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory    = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory  = np.zeros(self.mem_size, dtype=np.bool)

    def store_transitions(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index]     = self.new_state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]    = reward
        self.action_memory[index]    = action
        self.terminal_memory[index]  = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            # Take best action
            
            state = torch.tensor([observation]).to(self.Q_eval.device)
            if len(state.shape) == 2:
                state = state.unsqueeze(1) # give colour channel
            if len(state.shape) == 3:
                state = state.unsqueeze(1) # give batch id
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def learn(self):
        # Our initial memory is zero, so let's start learning when we have filled up at least one batch_size of experience
        if self.mem_cntr<self.batch_size:
            return
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        if self.modelname == "CNN":
          state_batch = state_batch.unsqueeze(1)
          new_state_batch = new_state_batch.unsqueeze(1)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min




def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)