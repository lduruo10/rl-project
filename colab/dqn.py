import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PONG = ['Pong-v0', 'Pong-v4']
CARTPOLE = ['CartPole-v0']

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = np.array([None] * capacity)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    def push(self, record):
        self.memory[self.position] = record
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
           (obs, action, next_obs, reward)
        """
        index_list = np.random.randint(self.size, size=batch_size)
        return zip(*self.memory[index_list])
    
class DQN(nn.Module):
    def __init__(self, env_config, env):
        super(DQN, self).__init__()
        
        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.eps_last = self.eps_start
        self.anneal_length = env_config["anneal_length"]
        self.epsilon_decay = (self.eps_start - self.eps_end)/self.anneal_length
        self.n_actions = env_config["n_actions"]
        self.env = env

        if self.env in CARTPOLE:
          self.fc1 = nn.Linear(4, 256)
          self.fc2 = nn.Linear(256, 64)
          self.fc3 = nn.Linear(64, 8)
          self.fc4 = nn.Linear(8, self.n_actions)
        elif self.env in PONG:
          self.conv1 = nn.Conv2d(env_config['obs_stack_size'], 32, kernel_size=8, stride=4, padding=0)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
          self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
          self.fc1 = nn.Linear(3136, 512)
          self.fc2 = nn.Linear(512, self.n_actions)
        else:
          raise Exception(self.env)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        if self.env in CARTPOLE:
          x = self.relu(self.fc1(x))
          x = self.relu(self.fc2(x))
          x = self.relu(self.fc3(x))
          x = self.fc4(x)
        elif self.env in PONG:
          x = self.relu(self.conv1(x))
          x = self.relu(self.conv2(x))
          x = self.relu(self.conv3(x))
          x = self.relu(self.fc1(self.flatten(x)))
          x = self.fc2(x)

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # TODO: Implement epsilon-greedy exploration.

        # Annealing
        if self.eps_last > self.eps_end:
            self.eps_last = self.eps_last - self.epsilon_decay

        # epsilon-greedy logic
        if not exploit:
            if random.random() < self.eps_last:
                return torch.tensor([[random.randrange(self.n_actions)]],device=device,dtype=torch.int64)

        with torch.no_grad():
            return self(observation).max(1)[1].view(1, 1).detach()


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return
   
    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    m_obs, m_action, m_next_obs, m_reward, m_done = memory.sample(dqn.batch_size)
 
    obs = torch.cat(m_obs).to(device).float()
    next_obs = torch.cat(m_next_obs).to(device).float()
    action = torch.cat(m_action).to(device).long()
    reward = torch.cat(m_reward).to(device).float()
    done = torch.cat(m_done).to(device).int()

    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.
    q_values = dqn(obs).gather(1, action)
   
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    q_targets_update = target_dqn(next_obs).detach().max(1)[0].unsqueeze(1)
    
    # Compute the target Q values, the last part will be 0 for terminal transitions
    q_value_targets = reward + (dqn.gamma * q_targets_update * (1 - done))
  
    # Compute loss.
    loss = F.mse_loss(q_values, q_value_targets)
    
    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    return loss.item()
