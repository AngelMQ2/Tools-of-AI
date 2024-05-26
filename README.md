# Tools-of-AI

This repository contains 3 artificial intelligence tools to solve an action-reward based problem. The implementations are related to the paper titled "Evolution vs. Learning: A Battle for Human-like Locomotion Control" by Ángel Manuel Montejo Quesada.
## Overview

The paper explores methods for achieving human-like locomotion control in robotic development. Specifically, it compares two Artificial Intelligence (AI) approaches: NeuroEvolution of Augmented Topologies (NEAT) and two variants of the Actor-Critic model.

## Folders

### NEAT

The NEAT folder contains the implementation of the NeuroEvolution of Augmented Topologies algorithm. NEAT is a Genetic Algorithm (GA) that evolves neural networks with non-fixed topology. It aims to optimize the performance of the network for locomotion control tasks.

### Custom A2C

The Custom A2C folder contains a customized version of the Actor-Critic algorithm for continuous action spaces. This implementation uses gradient methods to learn the action policy and approximates both the policy and value functions using Deep Neural Networks (DNNs).

### StableBaseLine3

The StableBaseLine3 folder contains the implementation of an optimized version of the Actor-Critic algorithm known as Asynchronous Advantage Actor-Critic (A3C). This variant uses multiple environments running in parallel to train a single model and aims to improve efficiency and convergence.

## Experiment Results

The repository includes experimental results comparing the performance and efficiency of NEAT against the two implementations of the Actor-Critic algorithm. While NEAT shows faster learning during training due to its exploration capabilities, the final performance of NEAT and A3C is statistically similar.

## Usage

Each folder contains its respective implementation. Users can explore the code, run experiments, and modify parameters as needed. Detailed instructions for running experiments are provided within each folder.

## Acknowledgements

We would like to acknowledge the contributions of the open-source community and the authors of the algorithms implemented in this repository. Special thanks to the YouTube community for their educational content on Artificial Intelligence algorithms.
