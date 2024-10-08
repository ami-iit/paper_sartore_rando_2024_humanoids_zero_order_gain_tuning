from dataBaseFitnessFunction import DatabaseFitnessFunction
import os
import pygad 
import pickle
import xml.etree.ElementTree as ET
import numpy as np 
from datetime import timedelta
import time 
import datetime
from utils_genetic import compute_fitness_pygad_with_torque,compute_fitness_pygad_without_torque, on_generation, get_initial_population,limit_cpu_cores, compute_fitness, initilize_directories
from include.geneticUilities.gaSettings import GASettings 


N_CORE = 100
ga_settings = GASettings()
bounds_ik = ga_settings.get_limits_IK_pygad()
bounds_mpc = ga_settings.get_limits_mpc_pygad()

bound = bounds_ik + bounds_mpc
limit_cpu_cores(N_CORE)


## Without torque 
for n_ in range(10):
    name_directory = "without_torque_" + str(n_)
    initilize_directories(name_directory)
    ga_instance = pygad.GA(
    num_generations=300,
    num_parents_mating=100, #10 
    sol_per_pop=100,
    fitness_func=compute_fitness_pygad_without_torque,
    gene_space=bound,
    num_genes=ga_settings.get_gene_space(), 
    parent_selection_type="tournament",
    K_tournament=4,
    crossover_type="two_points", # double point 
    allow_duplicate_genes=True,
    on_generation=on_generation,
    parallel_processing=['process', N_CORE], 
    keep_elitism = 10, 
    mutation_type="random"
    )

    ga_instance.run()

## With torque
for n_ in range(10):
    name_directory = "with_torque_" + str(n_)
    initilize_directories(name_directory)
    ga_instance = pygad.GA(
    num_generations=300,
    num_parents_mating=100, #10 
    sol_per_pop=100,
    fitness_func=compute_fitness_pygad_with_torque,
    gene_space=bound,
    num_genes=ga_settings.get_gene_space(), 
    parent_selection_type="tournament",
    K_tournament=4,
    crossover_type="two_points", # double point 
    allow_duplicate_genes=True,
    on_generation=on_generation,
    parallel_processing=['process', N_CORE], 
    keep_elitism = 10, 
    mutation_type="random"
    )

    ga_instance.run()
