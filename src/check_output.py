import numpy as np 
import pickle
from utils_genetic import compute_fitness

path = "ex_data/pop_ex.p"
opt = pickle.load(open( path, "rb" ) )
visualize_robot_flag = False
save_in_database = False
value = compute_fitness(opt[0], save_in_database=save_in_database, visualize=visualize_robot_flag)
