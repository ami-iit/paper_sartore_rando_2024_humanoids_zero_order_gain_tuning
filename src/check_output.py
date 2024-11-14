import numpy as np 
import pickle
from utils_genetic import compute_fitness
import argparse

parser = argparse.ArgumentParser(description="Check the output of the gain tuning procedure.")
parser.add_argument("--visualize", action="store_true", help="Set this flag to enable visualization.")
args = parser.parse_args()

# Set the visualize_robot_flag based on the argument
visualize_robot_flag = args.visualize

path = "ex_data/pop_ex.p"
opt = pickle.load(open( path, "rb" ) )
save_in_database = False
value = compute_fitness(opt[0], save_in_database=save_in_database, visualize=visualize_robot_flag)
