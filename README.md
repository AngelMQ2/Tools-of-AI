# Tools-of-AI
The idea behind this repository is to introduce non-expert users to Artificial Intelligence tools that have been discovered and can be applied to their own projects. It aims to showcase various AI approaches for environments with action-reward scenarios, providing both optimized implementations of modern tools and a custom version for more flexible modification and experimentation.

The implementations are related to the paper titled "Evolution vs. Learning: A Battle for Human-like Locomotion Control"  by Ángel Manuel Montejo Quesada, which aim a Human-like Locomotion Control for bipedal walking.
## Overview

The paper explores methods for achieving human-like locomotion control in robotic development. Specifically, it compares two Artificial Intelligence (AI) approaches: NeuroEvolution of Augmented Topologies (NEAT) and two variants of the Actor-Critic model.

## NEAT

The NEAT folder contains the implementation of the NeuroEvolution of Augmented Topologies algorithm using PyNEAT. NEAT is a Genetic Algorithm (GA) that evolves neural networks with non-fixed topology. It aims to optimize the performance of the network for locomotion control tasks.

References:
- [Evolving neural networks through augmenting topologies](https://www.mitpressjournals.org/doi/abs/10.1162/evco.2002.10.2.99) - Stanley, Kenneth O and Miikkulainen, Risto, Evolutionary Computation, 2002.

PyNEAT Documentation: [PyNEAT Documentation](https://pyneat.readthedocs.io/en/latest/)

## Custom A2C

The Custom A2C folder contains a customized version of the Actor-Critic algorithm for continuous action spaces. This implementation uses gradient methods to learn the action policy and approximates both the policy and value functions using Deep Neural Networks (DNNs).

## StableBaseLine3

The StableBaseLine3 folder contains the implementation of an optimized version of the Actor-Critic algorithm known as Asynchronous Advantage Actor-Critic (A3C). This variant uses multiple environments running in parallel to train a single model and aims to improve efficiency and convergence.

References:
- [Asynchronous methods for deep reinforcement learning](http://proceedings.mlr.press/v48/mniha16.html) - Mnih, Volodymyr et al., International Conference on Machine Learning, 2016.

Stable Baselines3 Documentation: [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/en/master/#)

## Experiment Results

The repository includes experimental results comparing the performance and efficiency of NEAT against the two implementations of the Actor-Critic algorithm. While NEAT shows faster learning during training due to its exploration capabilities, the final performance of NEAT and A3C is statistically similar.

## Usage

Each folder contains its respective implementation. Users can explore the code, run experiments, and modify parameters and environment as needed. Detailed instructions for running experiments are provided within each folder.

## Acknowledgements

We would like to acknowledge the contributions of the open-source community and the authors of the algorithms implemented in this repository. Special thanks to the YouTube community for their educational content on Artificial Intelligence algorithms.

