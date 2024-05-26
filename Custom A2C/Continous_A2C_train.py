import numpy as np
import random
import time
import os
from datetime import datetime

import gymnasium as gym
import math
import matplotlib
import matplotlib.pyplot as plt

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from tensorboardX import SummaryWriter

# Parameters:
HID_SIZE_VAL_1 = 200
HID_SIZE_VAL_2 = 64
GAMMA = 1            # Discount factor
BATCH_SIZE = 64      
ACTOR_LEARNING_RATE = 0.005 #5e-3 #0.001   
CRITIC_LEARNING_RATE = 0.001 #1e-3


# Tensorboard logfile writer:
log_path = "./LogFiles/"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
writer = SummaryWriter(log_dir=os.path.join(log_path, current_time))

model_path = "./Models/"
model_name = os.path.join(model_path,current_time)


# Create environment:
#env = gym.make("Humanoid-v4", terminate_when_unhealthy=True, forward_reward_weight= 1.75, render_mode = "human")
rand_noise = np.random.uniform(0,0.1)
env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, forward_reward_weight = 1.75)
#env = gym.make("Humanoid-v4")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

episode_durations = []
episode_reward = []
episode_time_alive = []

def save_logfiles(epoch):
    if len(episode_reward)>100:
        current_reward = episode_reward[-1]
        current_lenght = episode_time_alive[-1]
        # Average last 100 trials:
        mean_reward = sum(episode_reward[-100:])/len(episode_reward[-100:])
        mean_time_alive = sum(episode_time_alive[-100:])/len(episode_time_alive[-100:])

        writer.add_scalar('rollout/ep_rew_max', current_reward, epoch)
        writer.add_scalar('rollout/ep_rew_mean', mean_reward, epoch)
        writer.add_scalar('rollout/ep_len_mean', mean_time_alive, epoch)
        writer.add_scalar('rollout/ep_len_max', current_lenght, epoch)

def save_check_point(agent):
    # Directorio para guardar los puntos de control
    os.makedirs(model_path, exist_ok=True)

    # Guardar el modelo del actor
    torch.save(agent.actor.state_dict(), model_name)

    print(f"Modelo del actor guardado en {model_name}")


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def plot_reward(show_result=False, save=False):
    plt.figure(1)
    reward_t = torch.tensor(episode_reward, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(reward_t.numpy())
    # Take 100 episode averages and plot them too
    if len(reward_t) >= 100:
        means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    if save:
        plt.savefig('reward_plot.png')  # You can change the filename as needed

    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


########################## Replay memory buffet definition ############################
#######################################################################################

# Define transition to store: pair between (state,action) and (next_state, reward)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward','done'))

# Class consisting on a cyclic buffer of bounded size that holds the transitions observed recently
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    

#######################################################################################
############################# Actor Critic model ######################################

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dim, h_dim_1, h_dim_2, n_actions):
        super(GenericNetwork, self).__init__()

        # Hyper-parameters
        self.lr = lr
        self.input_dim = input_dim
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.n_actions = n_actions

        # Multilayer perceptron:
        self.layer1 = nn.Linear(self.input_dim, self.h_dim_1)
        self.layer2 = nn.Linear(self.h_dim_1, self.h_dim_2)
        self.layer2_2 = nn.Linear(self.h_dim_2, self.h_dim_2)
        self.bn = nn.BatchNorm1d(self.h_dim_2)
        self.layer3 = nn.Linear(self.h_dim_2, self.n_actions)  
        

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observations, training = False):
        state = torch.tensor(observations, dtype=torch.float32).to(self.device)

        # Forward pass:
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer2_2(x))
        if training:
            x = x.squeeze(1)  # Eliminar la dimensión intermedia de tamaño 1
            x = self.bn(x)
            #x = x.unsqueeze(1)  # Restaurar la dimensión eliminada
        
        x = self.layer3(x)  # Last layer does not have activation function cause it is handle afterward depending on it is critic or actor
       
        return x
    

class AgentA2C(nn.Module):
    def __init__(self, input_dim, n_actions, actor_lr, critic_lr, h_size_1 = 64, h_size_2 = 64, gamma = 0.99):
        super(AgentA2C, self).__init__()
        # Hyper-parameters:
        self.gamma = gamma
        self.log_prob = None    # Actor cost-function calculation
        self.n_outputs = n_actions
        self.n_actions = input_dim

        # Actor and critic NN:
        self.actor = GenericNetwork(actor_lr, input_dim=input_dim, n_actions=n_actions*2, h_dim_1=h_size_1, h_dim_2= h_size_2)
        self.critic = GenericNetwork(critic_lr, input_dim=input_dim, n_actions=1, h_dim_1=h_size_1, h_dim_2= h_size_2)
    
    def choose_action(self, observation, training = False):
        # Forward pass - handle dimension with and without batch
        if observation.dim() == 2:
            # Single observation with size [1, n_state]
            output = self.actor.forward(observation, training)
        elif observation.dim() == 3:
            # Batch of observations with size [batch_size, 1, n_state]
            observation = observation.squeeze(1)  # Remove the middle dimension
            output = self.actor.forward(observation, training)
        else:
            raise ValueError("Unsupported observation dimensions")

        # Extract mu and sigma of each action:
        mu = output[:,:self.n_outputs]
        std = output[:, self.n_outputs:]
        #std = torch.exp(std)    # Normalize std desviation ensure non-zero value

        # Activation function:
        mu = F.tanh(mu) 
        std = nn.Softplus()(std) + 1e-6 # Correction factor for numerical stability

        # Compute probability of taking an action:
        action_probs = torch.distributions.Normal(mu, std)
        # Sample an actual action from distribution:
        probs = action_probs.sample(sample_shape=torch.Size([1])).squeeze(0)  # Remove the leading dimension
        # Log probabilities:
        self.log_prob = action_probs.log_prob(probs)
        self.log_prob = torch.sum(self.log_prob,1).to(self.actor.device)

        # Clip the actions + activation function
        actions = F.tanh(probs)*0.4   # Concrete for the current environment

        return np.array(actions.tolist()[0]).reshape((-1,))

        

######################################################################################
############################## Optimize ##############################################

def optimize_model(agent):
    global memory

    if len(memory) < BATCH_SIZE:
        return

    # Set gradient to 0 - pytorch need this
    agent.actor.optimizer.zero_grad()
    agent.critic.optimizer.zero_grad()

    # Select random samples from memory:
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))  #This converts batch-array of Transitions to Transition of batch-arrays.

    # Sustract info:
    state_batch = tuple([s.tolist() for s in batch.state])
    action_batch = tuple([a.tolist() for a in batch.action])
    reward_batch = tuple([r.tolist() for r in batch.reward])
    done_batch = tuple([d.tolist() for d in batch.done])
    next_state_batch = tuple([[torch.zeros(agent.n_actions).tolist()] if s is None else s.tolist() for s in batch.next_state])

    # Convert to tensors:
    state_batch = torch.tensor(state_batch, dtype=torch.float32).to(agent.actor.device)
    action_batch = torch.tensor(action_batch, dtype=torch.float32).to(agent.actor.device)
    reward_batch = torch.tensor(reward_batch, dtype=torch.float32).to(agent.actor.device)
    next_state_batch = torch.tensor(next_state_batch, dtype=torch.float32).to(agent.actor.device)
    done_batch = torch.tensor(done_batch, dtype=torch.float32).to(agent.actor.device)

    # Actor forward pass - only to get log prob
    action_value = agent.choose_action(state_batch, training = True)

    # Critic forward pass:
    critic_value  = agent.critic.forward(state_batch, training = True)
    critic_value_next = agent.critic.forward(next_state_batch, training = True)

    # Temporal difference loss:
    delta = []
    for i,done in enumerate(done_batch):
        if done:
            delta.append(reward_batch[i] - critic_value[i])
        else:
            delta.append(reward_batch[i] + agent.gamma*critic_value_next[i] - critic_value[i])
    delta = torch.tensor(delta).to(agent.actor.device) # Convert delta list to a tensor

    # Computa actor & critic losses:
    actor_loss = torch.sum(-agent.log_prob * delta,0)
    critic_loss = torch.mean(delta**2,0)

    # Backpropagation:
    (actor_loss + critic_loss).backward()
    agent.actor.optimizer.step()
    agent.critic.optimizer.step()


####################################################################################
############################# Main Train Loop ######################################
# Define environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

# Initialize agent
agent = AgentA2C(input_dim = state_size,n_actions = action_size,
                 actor_lr = ACTOR_LEARNING_RATE, critic_lr = CRITIC_LEARNING_RATE,
                 h_size_1 = HID_SIZE_VAL_1, h_size_2 = HID_SIZE_VAL_2,
                 gamma = GAMMA).to(device)

# Instance memory:
memory = ReplayMemory(1000)

# Training parameters
EPISODES = 50000

for episode in range(EPISODES):
    state, _ = env.reset(seed=42)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    time_alive = 0
    ts = time.time()
    steps = 0

    # Life cycle
    for t in count():
        action = agent.choose_action(state)
        for element in action:
            if element is None or (isinstance(element, float) and math.isnan(element)):
                print('Wrong action value ', element, 'at epoch ', episode)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        time_alive += 1
        steps += 1
        reward = torch.tensor([reward], device=device)
        # End condition:
        done  = terminated or truncated
        done_ = torch.tensor([done], device=device)

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward, done_)

        # Learning process:
        optimize_model(agent)

        # Move to the next state
        state = next_state

        if done:
            if episode == EPISODES - 1:
                print("Test done is %.2f sec, reward %.3f, steps %d" % (time.time() - ts, total_reward, steps))
                save_check_point(agent)
            else:
                #print("Test done is %.2f sec, reward %.3f, steps %d" % (time.time() - ts, total_reward, steps), end = '\r')
                print(f"Epoch {episode}, reward {total_reward}. Test done is {time.time() - ts} sec, steps {steps}", end = '\r')
            episode_durations.append(t + 1)
            episode_time_alive.append(time_alive)
            episode_reward.append(total_reward)
            #plot_durations()
            #plot_reward()
            save_logfiles(episode)
            break
