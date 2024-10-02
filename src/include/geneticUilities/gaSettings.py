import numpy as np 

class GASettings:
    def __init__(self):
        self.string_low_limit = "low"
        self.string_upper_limit = "high"
        self.string_step = "step"
        self.define_limits_mpc_variables()
        self.define_limits_IK_parameters()



    def define_limits_mpc_variables(self):
        n_com_weigh = 2
        n_contact_position_weigth = 1
        n_force_rate_change_weigth = 3
        n_angular_momentum_weigth = 1
        n_contact_force_symmetry_weigth = 1
        self.n_mpc_variables = (
            n_com_weigh
            + n_contact_position_weigth
            + n_force_rate_change_weigth
            + n_angular_momentum_weigth
            + n_contact_force_symmetry_weigth
        )
        self.bound_com_x_y = [2.0, 50.0, 2.0]
        self.bound_com_z = [80, 140, 5.0]
        self.bound_general_mpc = [10.0, 150, 5]

    def get_limits_mpc_pygad(self):
        bounds = []
        bounds.append(
            self.get_dict_bound(
                self.bound_com_x_y[0], self.bound_com_x_y[1], self.bound_com_x_y[2]
            )
        )
        bounds.append(
            self.get_dict_bound(
                self.bound_com_z[0], self.bound_com_z[1], self.bound_com_z[2]
            )
        )
        for _ in range(self.n_mpc_variables - 2):
            bounds.append(
                self.get_dict_bound(
                    self.bound_general_mpc[0],
                    self.bound_general_mpc[1],
                    self.bound_general_mpc[2],
                )
            )
        return bounds
    
    def get_limits_mpc_array(self):
        bounds = self.get_limits_mpc_pygad()
        mpc_min = []
        mpc_max = []

        for item in bounds: 
            mpc_min.append(item[self.string_low_limit])
            mpc_max.append(item[self.string_upper_limit])
        return np.asarray(mpc_min), np.asarray(mpc_max)
    
    def get_dict_bound(self, min, max, step):
        if min > max:
            raise ValueError("Called set limits with min greater than max")
        if abs(max - min) < step:
            raise ValueError("Called set limits with too big step ")
        return {
            self.string_low_limit: min,
            self.string_upper_limit: max,
            self.string_step: step,
        }
 
    def define_limits_IK_parameters(self): 
        n_gains_com = 1 # Equal gain on x y 
        n_gains_zmp = 1 # Equal gain on x y 
        n_foot_linear = 1 
        n_foot_angular = 1 
        n_com_linear = 1
        n_chest_angular = 1 
        n_root_linear = 1 
        self.n_param_ik = n_gains_com + n_gains_zmp + n_foot_linear + n_foot_angular + n_com_linear + n_chest_angular + n_root_linear
        self.limit_linear_ik = [1.0,5.0, 0.2]
        self.limmit_foot_linear_ik = [2.5,5.0,0.2]
        self.limit_angular_ik = [1.0,10.0,0.4]
        self.limit_com_ik = [3.5,5.0,0.1]
        self.limit_zmp_ik = [0.5,1.0,0.1]

    def get_limits_IK_pygad(self):
        bounds = []
        bounds.append(self.get_dict_bound(self.limit_zmp_ik[0], self.limit_zmp_ik[1], self.limit_zmp_ik[2]))
        bounds.append(self.get_dict_bound(self.limit_com_ik[0],self.limit_com_ik[1],self.limit_com_ik[2]))
        bounds.append(self.get_dict_bound(self.limmit_foot_linear_ik[0], self.limmit_foot_linear_ik[1], self.limmit_foot_linear_ik[2]))
        bounds.append(self.get_dict_bound(self.limit_angular_ik[0], self.limit_angular_ik[1], self.limit_angular_ik[2]))
        bounds.append(self.get_dict_bound(self.limit_linear_ik[0], self.limit_linear_ik[1], self.limit_linear_ik[2]))
        bounds.append(self.get_dict_bound(self.limit_angular_ik[0], self.limit_angular_ik[1], self.limit_angular_ik[2]))
        bounds.append(self.get_dict_bound(self.limit_linear_ik[0], self.limit_linear_ik[1], self.limit_linear_ik[2]))
        return bounds
    
    def get_limits_IK_array(self):
        bounds = self.get_limits_IK_pygad()
        ik_min = []
        ik_max = []

        for item in bounds: 
            ik_min.append(item[self.string_low_limit])
            ik_max.append(item[self.string_upper_limit])
        return np.asarray(ik_min), np.asarray(ik_max)
    
    def get_gene_space(self): 
        return len(self.get_initial_guess())

    def get_initial_guess(self): 
        ## Definition of x_k
        # IK Parameters
        zmp_gain = 1.0
        com_gain = 3.5
        foot_linear = 3.0 
        foot_angular = 9.0
        com_linear = 2.0 
        chest_angular = 1.0
        root_linear = 2.0  
        ik_param = np.asarray([zmp_gain, com_gain, foot_linear, foot_angular, com_linear, chest_angular, root_linear])
        # MPC Parameters
        com_weight = np.asarray([100, 100])  # from 0-2 #from 26-28
        contact_position_weight = np.asarray([1e3]) / 1e2  # 3 #29
        force_rate_change_weight = np.asarray([10.0,10.0,10.0]) / 1e2  # from 4-6 #32
        angular_momentum_weight = np.asarray([1e5]) / 1e3  # 7 # 33
        contact_force_symmetry_weight = np.asarray([1.0]) / 1e2  # 34
        mpc_param = np.concatenate(
            (
                com_weight,
                contact_position_weight,
                force_rate_change_weight,
                angular_momentum_weight,
                contact_force_symmetry_weight,
            )
        )

        x_k = np.concatenate((ik_param, mpc_param))
        return x_k
