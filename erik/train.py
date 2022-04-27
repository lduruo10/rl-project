import argparse
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import torch
import os
import numpy as np
import config
from utils import preprocess,create_tensor
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--extend', type=int, default=0, help='Extend training of existing model')
parser.add_argument('--env', choices=['CartPole-v0','CartPole-v1','Pong-v0','Pong-v4'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole_v0,
    'CartPole-v1': config.CartPole_v1,
    'Pong-v0': config.Pong_v0,
    'Pong-v4': config.Pong_v4
}

PONG = ['Pong-v0', 'Pong-v4']  # Tried Pong-v4 when I couldn't run Pong-v0, but later Pong-v0 started working...
CARTPOLE = ['CartPole-v0']

if __name__ == '__main__':
    args = parser.parse_args()

    print('Available CUDA count:',torch.cuda.device_count())
    if torch.cuda.device_count()>0:
        print(torch.cuda.get_device_name(0))

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    if args.env in PONG:
        env = AtariPreprocessing(env, screen_size=env_config['screen_size'], grayscale_obs=True, frame_skip=1, terminal_on_life_loss=True, noop_max=30)
        env = FrameStack(env, env_config['obs_stack_size'])
        
    torch.set_default_dtype(torch.float32)

    # Initialize deep Q-networks. 
    if args.extend != 0:
        filename = "models/" + args.env + ".pt"
        if not os.path.isfile(filename):
            raise Exception("Model file does not exist: " + filename)
        dqn = torch.load(filename, map_location=device)
    else:
        dqn = DQN(env_config, args.env).to(device).float()

    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config, args.env).to(device).float()

    # Clone the original network, in case we loaded a model from disk
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    if args.extend == 0:
        with open("training_logs/" + args.env + ".log", 'w', encoding='utf-8') as f:
            f.write("episode,mean_return,mean_loss,epsilon,step_per_episode\n")

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    steps_total = 1

    for episode in range(1, env_config['n_episodes']+1):
        done = False
        e_return = 0
        steps_episode = 1
        
        obs = preprocess(env.reset(), args.env)
        obs_shape = obs.shape
        
        while not done:
            
            # TODO: Get action from DQN.
            action = dqn.act(obs, exploit=False)

            if args.env in PONG:         # As recommended, there are two actions for Pong in config.py (0 and 1)
                a_mapped = action.item() + 2   # The output from DQN will be 0 or 1, which is translated to 2 or 3 here
            elif args.env in CARTPOLE:
                a_mapped = action.item()
            else:
                raise Exception(args.env)

            # Act in the true environment.
            next_obs, reward, done, info = env.step(a_mapped)
            e_return += reward

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, args.env)
            else:
                next_obs = create_tensor(np.zeros(obs_shape))
              
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            memory.push((obs, action, next_obs, create_tensor([[reward]]), create_tensor([[int(done)]],t=torch.int32)))

            # Move to the next state
            obs = next_obs

            if len(memory) > env_config['replay_start_size']:
                # TODO: Run DQN.optimize() every env_config["train_frequency"] steps.
                if steps_total % env_config['train_frequency'] == 0:
                    
                    optimize(dqn, target_dqn, memory, optimizer)
                    
                # TODO: Update the target network every env_config["target_update_frequency"] steps.
                if steps_total % env_config['target_update_frequency'] == 0:
                    target_dqn.load_state_dict(dqn.state_dict())
                    target_dqn.eval()

            steps_episode += 1
            steps_total += 1

        if (args.env in PONG and episode % 5 == 0) or (args.env in CARTPOLE and episode % 20 == 0):   # Pong training is VERY slow
            print("Episode:", episode, "of", env_config["n_episodes"], ":", e_return, "steps:", steps_episode)

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print("Evaluation after episode", episode, ":", mean_return)

            with open("training_logs/" + args.env + ".log", 'a', encoding='utf-8') as f:
                f.write('f{episode},{mean_return},{dqn.eps_last},{steps_episode}\n')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                
                print('Best performance so far! Saving model.')
                torch.save(dqn, "models/" + args.env + ".pt")

    # Close environment after training is completed.
    env.close()
    
