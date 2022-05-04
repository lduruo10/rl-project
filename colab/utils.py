import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PONG = ['Pong-v0', 'Pong-v4']
CARTPOLE = ['CartPole-v0']

def preprocess(obs, env):
    """Performs necessary observation preprocessing."""
    if env in PONG:
        obs = np.array(obs)
        obs = obs / 255.0   # Scaling as recommended in instructions
        return torch.tensor(obs, device=device).unsqueeze(0).float()
    elif env in CARTPOLE:
        return torch.tensor(obs, device=device).unsqueeze(0).float()
    else:
        raise ValueError('Unknown environment:' + env)

def create_tensor(item, t=torch.float):
    return torch.tensor(item,device=device,dtype=t)
