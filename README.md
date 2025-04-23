# README for IoV Optimization Project

## Project Overview

This project aims to optimize latency and energy consumption in the Internet of Vehicles (IoV) using three different optimization approaches: **DDPG** (Deep Deterministic Policy Gradient), **NSGA-II** (Non-dominated Sorting Genetic Algorithm II), and **TD3** (Twin Delayed Deep Deterministic Policy Gradient). Each optimization method independently runs experiments that balance latency and energy consumption in IoV systems. The `compute_average_energy_and_latency.py` script is called by each approach to evaluate and report performance.

## Project Structure

This repository contains the following Python scripts:

1. **`compute_average_energy_and_latency.py`**  
   This script computes the average energy consumption and latency for the IoV system, providing key performance metrics. It is called by each optimization algorithm to calculate and assess the results.

2. **`DDPG_main.py`**  
   Implements the **DDPG** reinforcement learning algorithm to optimize latency and energy consumption. The script trains a deep reinforcement learning model to minimize these metrics. The output includes:

   - Latency and energy consumption curves over time.
   - Unit-time solution counts and diversity metrics.

3. **`NGSA_main.py`**  
   Implements the **NSGA-II** genetic algorithm to optimize the IoV system. This multi-objective approach evaluates trade-offs between latency and energy, generating solutions that form the Pareto front. The output includes:

   - The Pareto front showing latency and energy trade-offs.
   - Unit-time solution counts and diversity metrics.

4. **`TD3_main.py`**  
   Implements the **TD3** reinforcement learning algorithm, an improved version of DDPG. It focuses on stabilizing training and improving exploration. The output includes:

   - Latency and energy consumption curves over time.

## Model Overview

The optimization of latency and energy consumption is modeled as a mathematical problem that integrates communication, sensing, and computation components within IoV systems. Each optimization method (DDPG, NSGA-II, TD3) addresses this problem with distinct methodologies:

- **DDPG**: A deep reinforcement learning approach using continuous action spaces to optimize system performance.
- **NSGA-II**: A genetic algorithm that produces a Pareto front by evaluating the trade-offs between latency and energy consumption.
- **TD3**: A variant of DDPG that improves exploration by addressing Q-value overestimation, providing a more stable optimization process.

Each optimization method runs independently and utilizes `compute_average_energy_and_latency.py` to evaluate and output results.

## Results and Outputs

Each experiment produces the following outputs:

1. **NSGA-II**:
   - **Pareto Front**: A set of non-dominated solutions representing the trade-offs between latency and energy consumption.
   - **Unit-Time Solution Count**: The number of solutions that were feasible within each unit time.
   - **Diversity**: The variety of solutions generated over the course of the optimization process.

2. **DDPG**:
   - **Latency and Energy Curves**: The evolution of latency and energy consumption over time during training.
   - **Unit-Time Solution Count**: The number of solutions per unit time.
   - **Diversity**: The diversity of solutions achieved by the agent across episodes.

3. **TD3**:
   - **Latency and Energy Curves**: The evolution of latency and energy consumption over time during training.

Each experiment calls `compute_average_energy_and_latency.py` to compute and output the system’s overall performance, including energy and latency metrics.

## Requirements

To run the project, you need the following Python libraries:

- numpy
- torch
- matplotlib
- gym (if using reinforcement learning environments)
- scikit-learn
- pandas

You can install these dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

### 1. **Run NSGA-II Optimization**  
Execute the NSGA-II optimization by running:

```bash
python NGSA_main.py
```

This script will execute the NSGA-II genetic algorithm and generate solutions that optimize latency and energy consumption. The output includes the Pareto front and diversity metrics.

### 2. **Run DDPG Optimization**  
Execute the DDPG optimization using:

```bash
python DDPG_main.py
```

This script will train a reinforcement learning model using DDPG to minimize latency and energy consumption. The output includes latency and energy curves, as well as unit-time solution counts and diversity metrics.

### 3. **Run TD3 Optimization**  
Execute the TD3 optimization with:

```bash
python TD3_main.py
```

This script performs similar tasks as DDPG but uses TD3 to improve training stability and exploration. The output includes latency and energy curves.

### 4. **Compute Energy and Latency**  
Use the following script to compute the average energy consumption and latency after running the optimizations:

```bash
python compute_average_energy_and_latency.py
```

This script is called by each optimization algorithm to calculate and display the system's performance metrics.
