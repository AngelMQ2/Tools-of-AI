import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# Define la clase de la red neuronal genérica
class GenericNetwork(nn.Module):
    def __init__(self, input_dim, h_dim_1, h_dim_2, n_actions, device = 'cpu'):
        super(GenericNetwork, self).__init__()

        # Hyper-parameters
        #self.lr = lr
        self.device = device
        self.input_dim = input_dim
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.n_actions = n_actions

        # Multilayer perceptron:
        self.layer1 = nn.Linear(self.input_dim, self.h_dim_1)
        self.layer2 = nn.Linear(self.h_dim_1, self.h_dim_2)
        self.layer2_2 = nn.Linear(self.h_dim_2, self.h_dim_2)
        self.bn = nn.BatchNorm1d(self.h_dim_2)
        self.layer3 = nn.Linear(self.h_dim_2, 2*self.n_actions)  
        
    def choose_action(self, observation, training = False):
        # Forward pass - handle dimension with and without batch
        output = self.forward(observation, training)

        # Extract mu and sigma of each action:
        mu = output[:,:self.n_actions]
        std = output[:, self.n_actions:]
        #std = torch.exp(std)    # Normalize std desviation ensure non-zero value

        # Activation function:
        mu = F.tanh(mu)
        std = nn.Softplus()(std) + 1e-6 # Correction factor for numerical stability
        
        # Compute probability of taking an action:
        action_probs = torch.distributions.Normal(mu, std)
        # Sample an actual action from distribution:
        probs = action_probs.sample(sample_shape=torch.Size([1])).squeeze(0)  # Remove the leading dimension

        # Clip the actions + activation function
        actions = F.tanh(probs)*0.4   # Concrete for the current environment

        return np.array(actions.tolist()[0]).reshape((-1,))

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

def load_model(model_path):
    # Instancia la red neuronal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenericNetwork(input_dim=state_size, h_dim_1=HID_SIZE_VAL_1, h_dim_2=HID_SIZE_VAL_2, n_actions=action_size, device=device).to(device)
    # Carga los pesos guardados en el modelo
    model.load_state_dict(torch.load(model_path))
    # Establece el modo de evaluación (sin entrenamiento)
    model.eval()
    return model

def test_model(model, count=30):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rewards_list = []
    steps_list = []
    for _ in range(count):
        # Generate new env:
        rand_noise = np.random.uniform(0,0.1)
        test_env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, forward_reward_weight = 1.75, render_mode = "human")

        obs, info = test_env.reset()

        rewards = 0
        steps = 0
        while True:
            # Convierte la observación en tensor
            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # Pasa la observación a través del modelo para obtener la acción
            action_values = model.choose_action(state)
            # Extrae la primera mitad de las salidas como acciones
            #action = action_values[0, :action_size].cpu().detach().numpy()
            obs, reward, terminated, truncated, _ = test_env.step(action_values)
            done  = terminated or truncated
            rewards += reward
            steps += 1
            if done:
                test_env.close()
                break

        rewards_list.append(rewards)
        steps_list.append(steps)

    return rewards_list, steps_list

if __name__ == "__main__":
    # Parámetros
    HID_SIZE_VAL_1 = 200
    HID_SIZE_VAL_2 = 64

    # Carga el entorno de prueba
    rand_noise = np.random.uniform(0,0.1)
    test_env = gym.make("Humanoid-v4", reset_noise_scale = rand_noise, forward_reward_weight = 1.75, render_mode = "human")

    # Obtiene las dimensiones de entrada y salida del entorno
    state_size = test_env.observation_space.shape[0]
    action_size = test_env.action_space.shape[0]

    # Ruta al modelo guardado
    model_path = "C:/Users/angel/OneDrive/Documentos/Projects_AI/ToolsAI/Custom A2C/Models/2024-05-22_15-40"  # Ruta a ajustar según sea necesario

    # Carga el modelo
    actor_model = load_model(model_path)

    # Prueba el modelo
    reward_list, step_list = test_model(actor_model,count=500)

    np.save('./A2C_Stats',np.array(reward_list))

    # Plot the result
    # Fitness histogram
    plt.figure(figsize=(8, 6))  # Ancho x Alto en pulgadas
    plt.hist(reward_list, bins=20, color='blue', alpha=0.7)
    plt.ylabel('Fitness Value')
    plt.title('Histogram of Fitness Value')

    # Time alive histogram
    plt.figure(figsize=(8, 6))  # Ancho x Alto en pulgadas
    plt.hist(step_list, bins=20, color='blue', alpha=0.7)
    plt.ylabel('Time Alive')
    plt.title('Histogram of Time Alive')
    plt.show()