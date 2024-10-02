import os
import sys
import numpy as np
from tqdm import tqdm
import nevergrad as ng
from utils_genetic import compute_fitness_optimizer_minimize, compute_fitness_optimizer_with_torque_minimize, GASettings
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor as PPE, as_completed

from dataBaseFitnessFunction import DatabaseFitnessFunction


settings = GASettings()


mpc_min, mpc_max = settings.get_limits_mpc_array()
ik_min, ik_max = settings.get_limits_IK_array()

mpc_limits = np.array(list(zip(mpc_min, mpc_max))).reshape(-1, 2)
ik_limits = np.array(list(zip(ik_min, ik_max))).reshape(-1, 2)

bounds = np.vstack((ik_limits, mpc_limits))

d = bounds.shape[0]


# no_torque = int(sys.argv[1]) == 0
# num_workers = int(sys.argv[2])
# num_reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1

T = 300 # number of generations
popsize = 100 # population size
num_workers = 100 
num_reps = 10 

no_torque = True
for i in range(num_reps):

    name = f"cmaes_{i}"

    name = name + "_no_torque" if no_torque else name + "_with_torque"

    data_base = DatabaseFitnessFunction(name)
    data_base.create_empty_csv_database()
    num_evals = 0

    def print_candidate_and_value(optimizer, candidate, value):
        data_base.update([x for x in candidate.value], value, candidate.generation)
    x0 = np.mean(bounds, axis=1)
    x0 = ng.p.Array(init=x0,  lower=bounds[:, 0], upper=bounds[:, 1])

    cmaes = ng.optimizers.ParametrizedCMA(popsize=popsize,random_init=True,scale=10.0,elitist=False).set_name(f"CMAES_{i}", register=False)

# ng.optimizers.registry[f"CMAES_{i}"]
    optimizer = cmaes(parametrization=x0, budget=T * popsize, num_workers=num_workers)
    optimizer.register_callback("tell", print_candidate_and_value)

    if no_torque:
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
                recommendation = optimizer.minimize(compute_fitness_optimizer_minimize,executor=executor)
                print(recommendation.value)
    else:
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
                recommendation = optimizer.minimize(compute_fitness_optimizer_with_torque_minimize,executor=executor)
                print(recommendation.value)

no_torque = False
for i in range(num_reps):

    name = f"cmaes_{i}"

    name = name + "_no_torque" if no_torque else name + "_with_torque"

    data_base = DatabaseFitnessFunction(name)
    data_base.create_empty_csv_database()
    num_evals = 0

    def print_candidate_and_value(optimizer, candidate, value):
        data_base.update([x for x in candidate.value], value, candidate.generation)
    x0 = np.mean(bounds, axis=1)
    x0 = ng.p.Array(init=x0,  lower=bounds[:, 0], upper=bounds[:, 1])

    cmaes = ng.optimizers.ParametrizedCMA(popsize=popsize,random_init=True,scale=10.0,elitist=False).set_name(f"CMAES_{i}", register=False)

    optimizer = cmaes(parametrization=x0, budget=T * popsize, num_workers=num_workers)
    optimizer.register_callback("tell", print_candidate_and_value)

    if no_torque:
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(compute_fitness_optimizer_minimize,executor=executor)
            print(recommendation.value)
    else:
        with futures.ProcessPoolExecutor(max_workers=optimizer.num_workers) as executor:    
            recommendation = optimizer.minimize(compute_fitness_optimizer_with_torque_minimize,executor=executor)
            print(recommendation.value)
