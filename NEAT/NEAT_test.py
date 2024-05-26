import gymnasium as gym
import neat
import os
import neat.nn.feed_forward
import numpy as np
import matplotlib.pyplot as plt
import time

class WalkAgent:
    def __init__(self, test = False, plot=False):
        if test:
            rand_noise = np.random.uniform(0,0.1)
            self.env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise)
            if plot:
                self.env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, render_mode="human")
        else:
            if plot:
                self.env = gym.make("Humanoid-v4", render_mode="human")
            else:
                self.env = gym.make("Humanoid-v4")

        observation, info = self.env.reset(seed=42)
        self.state = observation
        self.reward = 0
        self.is_alive = True
        self.time_alive = 0
        self.goal = False
        self.show = plot

    def one_step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.time_alive += 1

        if terminated or truncated:
            observation, info = self.env.reset()

        self.state = observation
        self.reward = reward
        self.is_alive = not terminated
        self.goal = truncated

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

def find_best_genome(population,config):
    max_score = -np.inf
    nets = []
    agents = []
    genomes = []
    winner = None

    for _ in range(3):
        for idx, genome in population.items():
            genomes.append(genome)
            net = neat.nn.FeedForwardNetwork.create(genome,config)
            nets.append(net)
            genome.fitness = 0
            agents.append(WalkAgent(plot=False))
        
        while True:
            for index, agent in enumerate(agents):
                if agent.get_alive():
                    output = nets[index].activate(agent.get_data())
                    agent.one_step(output)

            remain_agents = 0
            for i, agent in enumerate(agents):
                if agent.get_alive():
                    remain_agents += 1
                    genomes[i].fitness += agent.get_reward()

                    # Select best agent:
                    if max_score < genomes[i].fitness:
                        max_score = genomes[i].fitness
                        winner = genomes[i]

            if remain_agents == 0:
                break
        
        for agent in agents:
            agent.shut_down()

    return winner

def evaluate_agent(net,n_iter=500):
    fitness_list = []
    lenght_list = []
    for _ in range(n_iter):
        agent = WalkAgent(test=True, plot=False)
        fitness = 0
        while agent.get_alive():
            output = net.activate(agent.get_data())
            agent.one_step(output)
            fitness += agent.get_reward()
            agent.render()
        
        fitness_list.append(fitness)
        lenght_list.append(agent.get_time_alive())
        
        agent.shut_down()

    return fitness_list, lenght_list

# Set configuration file
current_directory = os.path.dirname(__file__)
file_path = os.path.join(current_directory, "config-feedforward.txt")
config_path = "config-feedforward.txt"
config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                            neat.DefaultSpeciesSet, neat.DefaultStagnation, file_path)

# Load the checkpoint
checkpoint_path = "./Models/model_name"  
p = neat.Checkpointer.restore_checkpoint(checkpoint_path)
population = p.population



best_genome = find_best_genome(population, config)

# Ensure the genome is valid
if best_genome is None:
   raise RuntimeError("No best genome found in the checkpoint")
else:
    print("GENOME ENCONTRADO")



# Create the neural network from the best genome
net = neat.nn.FeedForwardNetwork.create(best_genome, config)

# Evaluate the best genome
fitness, time_alive = evaluate_agent(net,n_iter=500)

np.save('./NEAT_Stats4',np.array(fitness))

# Plot the result
# Fitness histogram
print(fitness)
plt.figure(figsize=(8, 6))  # Ancho x Alto en pulgadas
plt.hist(fitness, bins=20, color='blue', alpha=0.7)
plt.ylabel('Fitness Value')
plt.title('Histogram of Fitness Value')

# Time alive histogram
plt.figure(figsize=(8, 6))  # Ancho x Alto en pulgadas
plt.hist(time_alive, bins=20, color='blue', alpha=0.7)
plt.ylabel('Time Alive')
plt.title('Histogram of Time Alive')
plt.show()

# Optionally, you can save or log these statistics as needed.
