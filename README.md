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

## Execution

To prepare the dataset:

- Create an account at https://www.renewables.ninja/ and then generate an individual API token.
- Set the variable `TOKEN_API_GENERATION_DATA` in `prepare_dataset.py` to your generated API token.
- `conda activate myenv`
- Run the `prepare_dataset.py` script to automatically download and pre-process the dataset.

To run the experiments:

- Update the variable `PATH_TO_GLPK_EXECUTABLE` in `experiments/config.py`so that the path points to the `glpsol`
  executable of your glpk installation.
  - e.g., `PATH_TO_GLPK_EXECUTABLE = "~/miniconda3/envs/myenv/bin/glpsol"`.
- You can define the algorithms that should be executed in this experiment run in `run_experiments.py`.
    - e.g., `baseline_algorithms = ['milp', 'seq_milp']` (will be run with perfect information).
    - e.g., `other_algorithms = ['microppo', 'ppo_c', 'ppo_d', 'dqn', 'rb_economic', 'rb_own']` (will be run with forecasts).
- Run the `run_experiments.py` script. Note that the code will run in parallel by default on *all* available cores.

To create the plots, as in the paper:

- Run the `visualize_results.py` script.
- By default, the script will use the results of the example experiments run (cf. `output/experiments/example/example-run/`). You might change the `run_id` and `run_date` parameter
  in `visualize_results.py` to use the
  results of your experiment runs.
- The plots will be saved as PDF files to `output/plots/`.

## Reproducibility of the results

To reproduce the experimental results from the paper, please set `seed=10` in `experiments/config.py` and consider
the following hyperparameters for the various algorithms and the micro-grid environment.

### Algorithm parameters

For the experiments, we use the default discount factor (`gamma=0.9`) for all deep reinforcement learning (DRL) based
approaches. We train these approaches for one epoch. For the further hyperparameters of MicroPPO and the DRL-based
baselines (B5-B7), see Table I-II. All the (non-default) algorithm
parameters are set in `experiments/config.py`.

#### Table I: Parameters of MicroPPO, B5(PPO_C) and B7(PPO_D)

|                | **Policy**<br/>(`policy`) | **Learning rate**<br/>`learning_rate` | **Explor. rate**<br/>(`ent_coef`) | **Value function coef.**<br/>(`vf_coef`) | **batch Size**<br/>(`batch_size`) | **Update freq.**<br/>(`n_steps`) | **Actor network**<br/>(`pi`) | **Critic network**<br/>(`vf`) | 
|----------------|:-------------------------:|:-------------------------------------:|:---------------------------------:|:----------------------------------------:|:---------------------------------:|:--------------------------------:|:----------------------------:|:-----------------------------:|
| **MicroPPO**   |      MicroPPOPolicy       |              8.5 x 10e-4              |                0.1                |                   0.5                    |                168                |                7                 |          *- BNA -*           |           *default*           |
| **B5 (PPO_C)** |         MlpPolicy         |              5.0 x 10e-4              |                0.1                |                   0.5                    |                168                |                7                 |           (64,64)            |            (32,32)            |
| **B7 (PPO_D)** |         MlpPolicy         |              5.0 x 10e-4              |                0.1                |                   0.5                    |                168                |                7                 |           (64,64)            |            (32,32)            |

**BNA - **B**ranched **N**etwork **A**rchitecture*

#### Table II: Parameters of B6 (DQN)

|              | **Policy**<br/>(`policy`) | **Learning rate**<br/>`learning_rate` | **Buffer size**<br/>(`buffer_size`) | **Update Freq. Target**<br/>(`target_update_interval`) | **Steps before learning**<br/>(`learning_starts`) | **Networks**<br/>(`net_arch`) | 
|--------------|:-------------------------:|:-------------------------------------:|:-----------------------------------:|:------------------------------------------------------:|:-------------------------------------------------:|:-----------------------------:|
| **B6 (DQN)** |         MlpPolicy         |              5.0 x 10e-4              |             1.0 x 10e+6             |                          168                           |                        24                         |           *default*           |

For the PPO-based approaches (i.e., MicroPPO, B5, an B7), we use the clipping parameter `epsilon=0.2`.

### Micro-grid parameters

As described in the paper, we also need to set the parameters of the micro-grid environment. Table III provides an
overview of the
battery's parameters used. Table IV shows further relevant parameters of the micro-grid. All these parameters are set in
`experiments/model_factory.py`.

#### Table III: Parameters of the Battery

| Parameter                   |               Value                |
|-----------------------------|:----------------------------------:|
| Nominal capacity (kWh)      | 1 per 1,000 kWh annual consumption |
| Max. SoC (%)                |                 90                 |
| Min. SoC (%)                |                 10                 |
| SoC at t = 0 (%)            |                 50                 |
| Max. charging rate (kWh)    |           0.5 x capacity           |
| Max. discharging rate (kWh) |           0.5 x capacity           |
| Charging efficiency (%)     |                 90                 |
| Discharging efficiency (%)  |                 90                 |

#### Table IV: Other parameters of the environment

| Parameter                               |                Value                 |
|-----------------------------------------|:------------------------------------:|
| Nominal capacity of the PV system (kWP) | 1.5 per 1,000 kWh annual consumption |
| Discount factor for selling prices (%)  |                  25                  |