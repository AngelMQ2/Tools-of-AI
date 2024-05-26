import gymnasium as gym
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import neat
from neat.reporting import ReporterSet
import os

from tensorboardX import SummaryWriter
from datetime import datetime
import neat.checkpoint

import numpy as np

class WalkAgent:
    def __init__(self, plot=False):
        rand_noise = np.random.uniform(0,0.1)
        if plot:
            self.env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, forward_reward_weight = 1.75, render_mode="human")
        else:
            self.env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, forward_reward_weight = 1.75)

        observation, info = self.env.reset(seed=42)
        self.state = observation
        self.reward = 0
        self.is_alive = True
        self.time_alive = 0
        self.goal = False
        self.show = plot

    def one_step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.time_alive += 1  # Increment time_alive for each step

        if terminated or truncated:
            observation, info = self.env.reset()

        self.state = observation
        self.reward = reward
        self.is_alive = not terminated
        self.goal = truncated

        self.render()

    def get_data(self):
        return self.state
    
    def get_reward(self):
        return self.reward

    def get_alive(self):
        return self.is_alive

    def get_time_alive(self):
        return self.time_alive
    
    def render(self):
        if self.show:
            self.env.render()

    def shut_down(self):
        self.env.close()

global winner 
winner = 1

def walk(genomes, config, plot = False):
    global winner, writer

    # Init NEAT
    nets = []
    agents = []

    for id, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0

        # Init my agent
        if id == winner:
            agents.append(WalkAgent(plot=plot))
        else:
            agents.append(WalkAgent())

    # Main loop
    global generation
    generation += 1
    max_score = 0

    while True:
        for index, agent in enumerate(agents):
            if agent.get_alive():
                output = nets[index].activate(agent.get_data())
                agent.one_step(output)

        remain_agents = 0
        for i, agent in enumerate(agents):
            if agent.get_alive():
                remain_agents += 1
                genomes[i][1].fitness += agent.get_reward()

                # Select best agent:
                if max_score < genomes[i][1].fitness:
                    max_score = genomes[i][1].fitness
                    winner = genomes[i][0]

        if remain_agents == 0:
            break

    # Compute mean values:
    time_alive = []
    fitness_list = []
    for agent in agents:
        time_alive.append(agent.get_time_alive())
    for id, g in genomes:
        fitness_list.append(g.fitness)

    mean_time_alive = np.mean(time_alive)
    max_time_alive = np.max(time_alive)
    mean_reward = np.mean(fitness_list)
    max_reward = np.max(fitness_list)

    writer.add_scalar('rollout/ep_rew_max', max_reward, generation)
    writer.add_scalar('rollout/ep_rew_mean', mean_reward, generation)
    writer.add_scalar('rollout/ep_len_mean', mean_time_alive, generation)
    writer.add_scalar('rollout/ep_len_max', max_time_alive, generation)

    for agent in agents:
        agent.shut_down()

    print("Winner ID: ", winner)







###############     MAIN    #########################
# Set configuration file
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, "config-feedforward.txt")
config_path = "config-feedforward.txt"
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, file_path)


# Configure save file path:
model_path = "./Models/"
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
model_name = "model_" + current_time+"_"
model_name = os.path.join(model_path, model_name)



log_path = "./LogFiles/"
log_name = current_time
print(log_name)
writer = SummaryWriter(log_dir=os.path.join(log_path, current_time))
generation = 0

n_steps = 5000    # Number of generations to run

# Create core evolution algorithm class
p = neat.Population(config)
# To restore from a checkpoint:
# p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-<generation>')

# Add reporter for fancy statistical result
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Add Checkpointer
checkpointer = neat.Checkpointer(generation_interval=n_steps, time_interval_seconds=None, filename_prefix = model_name)
p.add_reporter(checkpointer)

# Run NEAT and save checkpoints
p.run(walk, n_steps)



