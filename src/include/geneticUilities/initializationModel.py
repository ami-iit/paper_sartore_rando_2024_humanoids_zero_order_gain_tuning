import os 
import xml.etree.ElementTree as ET
import numpy as np 
from include.robotModel.robotModel import RobotModel
from dataBaseFitnessFunction import DatabaseFitnessFunction
class InitializationModel(): 

    def __init__(self) -> None:
        self.common_path = os.path.dirname(os.path.abspath(__file__))
        ## Defining the urdf path of both the startin, del and the modified one
        urdf_path_original = self.common_path + "/../../models/urdf/ergoCub/robots/ergoCubSN000/model.urdf"
        # World urdf path
        self.word_path = self.common_path + "/autogenerated/model_modified.world"
        # Load the URDF file
        tree = ET.parse(urdf_path_original)
        root = tree.getroot()
        
        # Convert the XML tree to a string
        self.robot_urdf_string_original = ET.tostring(root)
        ## Variable frequency 
        self.frequency_seconds_ik = 0.01
        self.frequency_milliseconds_centroidal_mpc = 100

    
    def create_directories(self,n_iter):
            self.common_path = os.path.dirname(os.path.abspath(__file__)) + "/../../"+ str(n_iter)
            os.mkdir(self.common_path)
            os.mkdir(self.common_path+"/results/")
            self.dataBase_instance = DatabaseFitnessFunction(
                self.common_path + "/results/fitnessDatabase"
            )
            
    def define_robot_model_with_torso(self): 
        self.with_torso = True
        joint_name_list = [
            "r_hip_pitch",#0
            "r_hip_roll",#1
            "r_hip_yaw",#2
            "r_knee",#3
            "r_ankle_pitch",#4
            "r_ankle_roll",#5
            "l_hip_pitch",#6
            "l_hip_roll",#7
            "l_hip_yaw",#8
            "l_knee",#9
            "l_ankle_pitch", #10
            "l_ankle_roll",#11
            "torso_pitch",#12
            "torso_roll",#13
            "torso_yaw",#14
            "r_shoulder_pitch", #15
            "r_shoulder_roll",#16
            "r_shoulder_yaw",#17
            "r_elbow",#18
            "l_shoulder_pitch",#19
            "l_shoulder_roll",#20
            "l_shoulder_yaw",#21
            "l_elbow"#22
        ]
        self.mujoco_path = self.common_path+ "/../../models/urdf/ergoCub/robots/ergoCubSN000/model_with_torso.xml"
        self.robot_model_init = RobotModel(self.robot_urdf_string_original, self.word_path, joint_name_list)
        self.s_des = np.array( [ 0.56056952, 0.01903913, -0.0172335, -1.2220763, -0.52832664, -0.02720832, 0.56097981, 0.0327311 ,-0.02791293,-1.22200495,  -0.52812215, -0.04145696, 0.12, 0.0, 0.0, 0.02749586, 0.25187149, -0.14300417, 0.6168618, 0.03145343, 0.25644825, -0.14427671, 0.61634549,])
        contact_frames_pose = {self.robot_model_init.left_foot_frame: np.eye(4),self.robot_model_init.right_foot_frame: np.eye(4)}
        H_b = self.robot_model_init.get_base_pose_from_contacts(self.s_des, np.zeros(self.robot_model_init.NDoF), contact_frames_pose) 
        self.xyz_rpy = np.zeros(6)
        self.xyz_rpy[:3] = H_b[:3,3]
        rpy = matrix_to_rpy(H_b[:3,:3])
        self.xyz_rpy[3:] = rpy 

    def define_robot_model_no_torso(self): 
        self.with_torso = False
        joint_name_list = [
            "r_hip_pitch",#0
            "r_hip_roll",#1
            "r_hip_yaw",#2
            "r_knee",#3
            "r_ankle_pitch",#4
            "r_ankle_roll",#5
            "l_hip_pitch",#6
            "l_hip_roll",#7
            "l_hip_yaw",#8
            "l_knee",#9
            "l_ankle_pitch", #10
            "l_ankle_roll",#11
            "r_shoulder_pitch", #12
            "r_shoulder_roll",#13
            "r_shoulder_yaw",#14
            "r_elbow",#15
            "l_shoulder_pitch",#16
            "l_shoulder_roll",#17
            "l_shoulder_yaw",#18
            "l_elbow"#19
        ]
        self.mujoco_path = self.common_path+ "/../../models/urdf/ergoCub/robots/ergoCubSN000/muj_model.xml"
        self.robot_model_init = RobotModel(self.robot_urdf_string_original, self.word_path, joint_name_list)
        self.s_des = np.array( [ 0.56056952, 0.01903913, -0.0172335, -1.2220763, -0.52832664, -0.02720832, 0.56097981, 0.0327311 ,-0.02791293,-1.22200495,  -0.52812215, -0.04145696, 0.02749586, 0.25187149, -0.14300417, 0.6168618, 0.03145343, 0.25644825, -0.14427671, 0.61634549,])
        contact_frames_pose = {self.robot_model_init.left_foot_frame: np.eye(4),self.robot_model_init.right_foot_frame: np.eye(4)}
        H_b = self.robot_model_init.get_base_pose_from_contacts(self.s_des, np.zeros(self.robot_model_init.NDoF), contact_frames_pose) 
        self.xyz_rpy = np.zeros(6)
        self.xyz_rpy[:3] = H_b[:3,3]
        rpy = matrix_to_rpy(H_b[:3,:3])
        self.xyz_rpy[3:] = rpy 

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
