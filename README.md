# MicroPPO: Safe Power Flow Management in Decentralized Micro-grids with Proximal Policy Optimization

This repository contains code for the paper *MicroPPO: Safe Power Flow Management in Decentralized Micro-grids with
Proximal Policy Optimization*, to appear in the Proceedings of the 11th IEEE International Conference on Data Science 
and Advanced Analytics, DSAA 2024.

## Abstract

Future sustainable energy systems require the integration of local renewable energy sources~(RES) into decentralized 
micro-grids, each containing RES, energy storage systems, and local loads. 
A substantial challenge associated with micro-grids is the optimization of energy flows to minimize operating costs. 
This is particularly complex due to (a) the fluctuating power generation of RES, (b) the variability of local loads, 
and (c) the possibility of energy trade between a micro-grid and a larger "utility grid" that it connects to. 
Existing methods struggle to manage these sources of uncertainty effectively.

To address this, we propose MicroPPO, a reinforcement learning approach for real-time management of power flows in such 
small-scale energy systems.
MicroPPO introduces a novel definition of the environment as a Markov Decision Process~(MDP) with a continuous and 
multi-dimensional action space.
This enables more precise control of power flows compared to discrete methods. Additionally, MicroPPO employs an 
innovative actor network architecture featuring multiple network branches to reflect the individual action dimensions. 
It further integrates a differentiable projection layer that enforces the feasibility of actions. 

We assess the performance of our approach against state-of-the-art methods using real-world data. 
Our results demonstrate MicroPPO's superior convergence towards near-optimal policies.

## Installation of dependencies

We recommend to create a new conda environment using Python version 3.7 to run our code. If Anaconda or Miniconda is not
yet installed, install it first.

To create a new conda environment, run

- `conda create -n "myenv" python=3.7`.

Then activate the environment and install the dependencies from the `requirements.txt` using

- `pip install -r requirements.txt`.

Unfortunately, due to some dependency issues, we must install `pyomo` and `glpk` package manually by running the
following commands:

- `conda install -c conda-forge pyomo`,
- `conda install -c conda-forge glpk`.

If you encounter any problems during installation please have a look at the packages listed in the file and install them
also manually.