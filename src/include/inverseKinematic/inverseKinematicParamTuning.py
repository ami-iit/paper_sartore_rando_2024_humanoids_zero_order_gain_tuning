import numpy as np 

class InverseKinematicParamTuning(): 
    
    def __init__(self) -> None:
        self.zmp_gain = [0.2,0.2]
        self.com_gain = [2.0,2.0]
        self.foot_linear = 3.0 
        self.foot_angular = 9.0
        self.com_linear = 2.0 
        self.chest_angular = 1.0
        self.root_linear = 2.0  

    def set_joint_weigth(self,with_torso): 
        if(with_torso): 
            self.weigth_joint = [1.0,1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0,2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0]
        else: 
            self.weigth_joint = [1.0,1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0, 1.0,2.0, 2.0, 2.0, 2.0,2.0, 2.0, 2.0, 2.0]
            

    def set_parameters(self, zmp_gain,com_gain, foot_linear, foot_angular, com_linear, chest_angular, root_linear): 
        self.zmp_gain = [zmp_gain,zmp_gain]
        self.com_gain = [com_gain,com_gain]
        self.foot_linear = foot_linear
        self.foot_angular = foot_angular
        self.com_linear = com_linear 
        self.chest_angular = chest_angular
        self.root_linear = root_linear 
    def set_from_xk(self,x_k): 
        #x_k[zmp_gain_1, zmp_gain_2, com_gain_1,com_gain_2, foot_linear, foot_angular, com_linear, chest_angular, root_linear]
        #dimension 9 
        self.set_parameters(zmp_gain=x_k[0], com_gain=x_k[1], foot_linear=x_k[2], foot_angular=x_k[3],com_linear= x_k[4], chest_angular=x_k[5], root_linear=x_k[6])

    def __str__(self):
        return (f"zmp_gain: {self.zmp_gain}, "
                f"com_gain: {self.com_gain}, "
                f"foot_linear: {self.foot_linear}, "
                f"foot_angular: {self.foot_angular}, "
                f"com_linear: {self.com_linear}, "
                f"chest_angular: {self.chest_angular}, "
                f"root_linear: {self.root_linear}")