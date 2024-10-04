from include.mujocoSimulator.mujocoSimulator import MujocoSimulator
from include.centroidalMPC.centroidalMPC import CentroidalMPC
from include.centroidalMPC.mpcParameterTuning import MPCParameterTuning
from include.inverseKinematic.inverseKinematicParamTuning import InverseKinematicParamTuning
from dataBaseFitnessFunction import DatabaseFitnessFunction
import os
import pickle
import xml.etree.ElementTree as ET
from include.inverseKinematic.inverseKinematic import inverseKinematic
import numpy as np 
from datetime import timedelta
from include.geneticUilities.initializationModel import InitializationModel
from include.geneticUilities.gaSettings import GASettings

def limit_cpu_cores(max_cores):
    # Set the maximum number of CPU cores
    os.environ["OMP_NUM_THREADS"] = str(max_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_cores)
    os.environ["MKL_NUM_THREADS"] = str(max_cores)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_cores)

def clip_input(x_k, min_param, max_param): 
    len_xk = len(x_k)
    for idx in range(len_xk): 
        el = x_k[idx]
        min_el = min_param[idx]
        max_el = max_param[idx]
        if(el<min_el): 
            x_k[idx] = min_el
        elif(el>max_el): 
            x_k[idx] = max_el
    return x_k 

def matrix_to_rpy(matrix):
    """Converts a rotation matrix to roll, pitch, and yaw angles in radians.

    Args:
        matrix (numpy.ndarray): 3x3 rotation matrix.

    Returns:
        tuple: Tuple containing the roll, pitch, and yaw angles in radians.
    """
    assert matrix.shape == (3, 3), "Input matrix must be a 3x3 rotation matrix."

    # Extract rotation angles from the rotation matrix
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    pitch = np.arctan2(-matrix[2, 0], np.sqrt(matrix[2, 1]**2 + matrix[2, 2]**2))
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])
    rpy = np.zeros(3)
    rpy[0] = roll
    rpy[1] = pitch
    rpy[2] = yaw
    return rpy

control_torso = False
initializer = InitializationModel()
if(control_torso):
    initializer.define_robot_model_with_torso()
else: 
    initializer.define_robot_model_no_torso()

def initilize_directories(n_iter):
    initializer.create_directories(n_iter)
    
def compute_fitness(x_k, save_in_database= True, visualize = False, weigths_tasks= [1.0, 0.0]):
    ga_settings_istance = GASettings()
    ik_min, ik_max = ga_settings_istance.get_limits_IK_array()
    mpc_min, mpc_max = ga_settings_istance.get_limits_mpc_array()
    min_param = np.concatenate((ik_min,mpc_min))
    max_param = np.concatenate((ik_max,mpc_max))
    x_k = clip_input(x_k, min_param, max_param)
    ik_gain = x_k[:ga_settings_istance.n_param_ik]
    mpc_gains = x_k[ga_settings_istance.n_param_ik:]
    if(save_in_database):
        fitness_value = initializer.dataBase_instance.get_fitness_value(chromosome=x_k)
        # if already computed fitness value returing the value in the database
        if fitness_value is not None:
            print("Already comptued fitness")
            return fitness_value
    # Defining parameters
    mpc_parameters = MPCParameterTuning()
    mpc_parameters.set_from_xk(mpc_gains)
    ik_parameters = InverseKinematicParamTuning()
    ik_parameters.set_from_xk(ik_gain)
    ik_parameters.set_joint_weigth(initializer.with_torso)

    # Defining IK instance
    IK_controller_instance = inverseKinematic(frequency=initializer.frequency_seconds_ik,robot_model=initializer.robot_model_init)
    IK_controller_instance.define_tasks(parameters=ik_parameters)

    # Defining MPC 
    mpc = CentroidalMPC(robot_model=initializer.robot_model_init,frequency_ms = initializer.frequency_milliseconds_centroidal_mpc,frequency_foot_position_seconds=initializer.frequency_milliseconds_centroidal_mpc/1000)

    # Defining the simulator 
    mujoco_instance = MujocoSimulator()
    mujoco_instance.load_model(initializer.robot_model_init, s= initializer.s_des, xyz_rpy=initializer.xyz_rpy, mujoco_path=initializer.mujoco_path)
    # mujoco_instance.set_simulation_time(0.0001)
    s,ds,tau = mujoco_instance.get_state()
    t = mujoco_instance.get_simulation_time()
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    
    IK_controller_instance.set_state_with_base(s,ds,H_b,w_b,t)
    
    n_step = int(IK_controller_instance.frequency/mujoco_instance.get_simulation_frequency())
    n_step_mpc_tsid = int(mpc.get_frequency_seconds()/IK_controller_instance.frequency)

    
    mpc.intialize_mpc(mpc_parameters=mpc_parameters)
    mpc.configure(s_init=initializer.s_des, H_b_init=H_b)
    IK_controller_instance.update_com(H_b, s)
    mpc.define_test_com_traj(IK_controller_instance.com)
    TIME_TH = 20
    mujoco_instance.set_visualize_robot_flag(visualize)
    mujoco_instance.step(1)
    s,ds,tau = mujoco_instance.get_state()
    
    H_b = mujoco_instance.get_base()
    w_b = mujoco_instance.get_base_velocity()
    t = mujoco_instance.get_simulation_time()
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
    mpc.initialize_centroidal_integrator(s=s, s_dot=ds,H_b=H_b, w_b=w_b,t=t)
    mpc_output = mpc.plan_trajectory()  

    # Update MPC and getting the state
    mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
    IK_controller_instance.set_state_with_base(s,ds,H_b,w_b,t)
    IK_controller_instance.update_com(H_b, s)
    IK_controller_instance.set_desired_base_orientation()
    counter = 0 
    mpc_success = True
    succeded_controller = True
    IK_controller_instance.define_integrator()
    norm_torque_list = []
    total_misalignment = 0.0
    H_b_ik = H_b
    w_b_ik = w_b
    while(t<TIME_TH):
        # print(t)
        robot_standing = True 
        # Reading robot state from simulator
        s,ds,tau = mujoco_instance.get_state()
        left_foot_contact, right_foot_contact = mujoco_instance.get_contact_status()

        norm_torque_list.append(np.linalg.norm(tau))
        H_b = mujoco_instance.get_base()

        if(H_b[2,3]<0.5): 
            robot_standing = False
            print("Robot felt")
            break
        
        w_b = mujoco_instance.get_base_velocity()
        t = mujoco_instance.get_simulation_time()
        
        if(counter == 0):
            mpc.set_state_with_base(s=s, s_dot=ds, H_b=H_b, w_b=w_b,t=t)
            mpc.update_references()
            mpc_success = mpc.plan_trajectory()
            mpc.contact_planner.advance_swing_foot_planner()        
       
            if(not(mpc_success)): 
                print("MPC failed")
                break
        
        com, dcom,forces_left, forces_right, ang_mom = mpc.get_references()
        left_foot, right_foot = mpc.contact_planner.get_references_swing_foot_planner()
        left_foot_front_wrench, left_foot_rear_wrench, rigth_foot_front_wrench, rigth_foot_rear_wrench = mujoco_instance.get_feet_wrench()
        IK_controller_instance.compute_zmp(left_foot_front_wrench, left_foot_rear_wrench, rigth_foot_front_wrench, rigth_foot_rear_wrench)
        IK_controller_instance.update_task_references_mpc(com=com, dcom=dcom,ddcom=np.zeros(3),left_foot_desired=left_foot, right_foot_desired=right_foot,s_desired=np.array(initializer.s_des), wrenches_left = forces_left, wrenches_right = forces_right, H_omega=ang_mom)
        succeded_controller = IK_controller_instance.run()
        IK_controller_instance.update_state()
        IK_controller_instance.update_com(H_b, s) 
        
        if(not(succeded_controller)): 
            print("Controller failed")
            break
        
        s_ctrl, H_b_ik = IK_controller_instance.get_output()
        
        if(control_torso):
            mujoco_instance.set_input(s_ctrl)
        else: 
            mujoco_instance.set_position_input(s_ctrl)

        mujoco_instance.step(n_step=n_step)
        feet_normal, misalignment_feet = mujoco_instance.check_feet_status(s, H_b,)
        total_misalignment += misalignment_feet
        counter = counter+ 1 

        if(counter == n_step_mpc_tsid): 
            counter = 0 

    # fitness_minimize = 12*total_misalignment + 100*(TIME_TH + mujoco_instance.get_simulation_frequency()-t) + 0.001*np.mean(norm_torque_list)
            
    fitness_minimize =weigths_tasks[0]*(TIME_TH + mujoco_instance.get_simulation_frequency()-t) + weigths_tasks[1]*(np.mean(norm_torque_list))
#    return fitness_minimize
    fitness = 1000*float(1/fitness_minimize)
    return fitness

def compute_fitness_pygad(pygadClass, x_k, idx): 
    fitness = compute_fitness_optimizer_with_torque_maximize(x_k)
    return fitness

def compute_fitness_optimizer_maximize(x_k): 
    weigths = [1.0, 0.0] # The first gain is the time difference, the second one the mean of torque norm 
    fitness = compute_fitness(x_k, False, False,weigths = [1.0, 0.0]) # The first gain is the time difference, the second one the mean of torque norm
    return fitness

def compute_fitness_optimizer_with_torque_maximize(x_k): 
    weigths = [100, 0.001] # The first gain is the time difference, the second one the mean of torque norm 
    fitness = compute_fitness(x_k, False,  False, weigths)
    return fitness

def compute_fitness_optimizer_minimize(x_k): 
    weigths = [1.0, 0.0] # The first gain is the time difference, the second one the mean of torque norm 
    fitness = compute_fitness(x_k, False,  False, weigths)
    fitness_min = - fitness 
    return fitness_min 

def compute_fitness_optimizer_with_torque_minimize(x_k): 
    weigths = [100, 0.001] # The first gain is the time difference, the second one the mean of torque norm 
    fitness = compute_fitness(x_k, False,  False, weigths)
    fitness_min = - fitness 
    return fitness_min 

def on_generation(ga_instance):

    common_path = os.path.dirname(os.path.abspath(__file__))
    dataBase_instance = DatabaseFitnessFunction(
    common_path + "/results/fitnessDatabase"
    )
    generation_completed = ga_instance.generations_completed
    population = ga_instance.population
    fitness = ga_instance.last_generation_fitness
    for indx in range(len(fitness)): 
        fitness_value = fitness[indx]
        x_k = population[indx]
        dataBase_instance.update(x_k, fitness_value,generation_completed)
    filename_i = common_path + "/results/genetic" + str(generation_completed)+ ".p"
    pickle.dump(ga_instance.population, open(filename_i, "wb"))

def get_initial_population(): 
    population_initial_old = pickle.load( open( "population_init.p", "rb" ) )
    population_initial = []
    for item in population_initial_old: 
        item_new = np.concatenate((item[:8], item[9:]))
        population_initial.append(item_new)
    return population_initial