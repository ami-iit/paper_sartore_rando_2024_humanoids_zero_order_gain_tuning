from adam.casadi.computations import KinDynComputations
import numpy as np
from urchin import URDF
from urchin import Joint
from urchin import Link
import mujoco
import tempfile
import xml.etree.ElementTree as ET
import idyntree.bindings as iDynTree

class RobotModel(KinDynComputations):
    def __init__(self, urdfstring: str, world_path:str , joint_name_list:list) -> None:
        self.urdf_string = urdfstring
        self.world_path = world_path
        self.robot_name = "icubSim"
        self.joint_name_list = joint_name_list
        self.right_arm_indexes = [0, 4]
        self.left_arm_indexes = [4, 8]
        self.right_leg_indexes = [8, 14]
        self.left_leg_indexes = [14, 20]
        self.remote_control_board_list = [
            "/" + self.robot_name + "/torso",
            "/" + self.robot_name + "/left_arm",
            "/" + self.robot_name + "/right_arm",
            "/" + self.robot_name + "/left_leg",
            "/" + self.robot_name + "/right_leg",
        ]

        self.base_link = "root_link"
        self.left_foot_frame = "l_sole"
        self.right_foot_frame = "r_sole"
        self.torso_link = "chest"
        legs_gain_kp = np.array([35*70.0, 35*70.0, 35*40.0, 35*100.0, 35*100.0, 35*100.0])
        arms_gain_kp = np.array([20*5.745, 20*5.745, 20*5.745, 20*1.745])
        torso_gain_kp = np.array([100,120,120])
        legs_gain_kd = np.array([15*0.15,15*0.15,15*0.35,15*0.15,15*0.15,15*0.15])
        arms_gain_kd = np.array([4*5.745,4*5.745,4*5.745,4*1.745])
        torso_gain_kd = np.array([10,10,10])
        self.kp_position_control = np.concatenate((legs_gain_kp, legs_gain_kp, arms_gain_kp, arms_gain_kp))
        self.kd_position_control = np.concatenate((legs_gain_kd, legs_gain_kd, arms_gain_kd,arms_gain_kd))
        self.ki_position_control = 0*self.kd_position_control
        self.mujoco_lines_urdf = '<mujoco> <compiler discardvisual="false"/> </mujoco>'
        self.gravity = iDynTree.Vector3()
        self.gravity.zero()
        self.gravity.setVal(2, -9.81)
        self.H_b = iDynTree.Transform()
        super().__init__(urdfstring, self.joint_name_list, self.base_link)
        self.H_left_foot = self.forward_kinematics_fun(self.left_foot_frame)
        self.H_right_foot = self.forward_kinematics_fun(self.right_foot_frame)
        
    def get_idyntree_kyndyn(self):
        model_loader = iDynTree.ModelLoader()
        model_loader.loadReducedModelFromString(self.urdf_string.decode('utf-8'), self.joint_name_list)
        kindyn = iDynTree.KinDynComputations()
        kindyn.loadRobotModel(model_loader.model())
        return kindyn

    def get_initial_configuration(self):

        base_position_init = np.array([-0.0489, 0, 0.65])
        base_orientationQuat_init = np.array([0, 0, 0, 1])
        base_orientationQuat_position_init = np.concatenate(
            (base_orientationQuat_init, base_position_init), axis=0
        )

        s_init = {
            "torso_pitch": -3,
            "torso_roll": 0,
            "torso_yaw": 0,
            "l_shoulder_pitch": -35.97,
            "l_shoulder_roll": 29.97,
            "l_shoulder_yaw": 0.006,
            "l_elbow": 50,
            "r_shoulder_pitch": -35.97,
            "r_shoulder_roll": 29.97,
            "r_shoulder_yaw": 0.006,
            "r_elbow": 50,
            "l_hip_pitch": 12,
            "l_hip_roll": 5,
            "l_hip_yaw": 0,
            "l_knee": -10,
            "l_ankle_pitch": -1.6,
            "l_ankle_roll": -5,
            "r_hip_pitch": 12,
            "r_hip_roll": 5,
            "r_hip_yaw": 0,
            "r_knee": -10,
            "r_ankle_pitch": -1.6,
            "r_ankle_roll": -5,
        }

        return [s_init, base_orientationQuat_position_init]

    def get_joint_limits(self):
        joints_limits = {
            "torso_pitch": [-0.3141592653589793, 0.7853981633974483],
            "torso_roll": [-0.4014257279586958, 0.4014257279586958],
            "torso_yaw": [-0.7504915783575618, 0.7504915783575618],
            "l_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "l_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "l_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "l_elbow": [0.3, 1.3089969389957472],
            "r_shoulder_pitch": [-1.53588974175501, 0.22689280275926285],
            "r_shoulder_roll": [0.20943951023931956, 2.8448866807507573],
            "r_shoulder_yaw": [-0.8726646259971648, 1.3962634015954636],
            "r_elbow": [0.3, 1.3089969389957472],
            "l_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "l_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "l_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "l_knee": [-1.2, -0.4],
            "l_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "l_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
            "r_hip_pitch": [-0.7853981633974483, 2.007128639793479],
            "r_hip_roll": [-0.17453292519943295, 2.007128639793479],
            "r_hip_yaw": [-1.3962634015954636, 1.3962634015954636],
            "r_knee": [-1.2, -0.4],
            "r_ankle_pitch": [-0.7853981633974483, 0.7853981633974483],
            "r_ankle_roll": [-0.4363323129985824, 0.4363323129985824],
        }
        return joints_limits

    def get_left_arm_from_joint_position(self, s):
        return np.array(s[self.left_arm_indexes[0] : self.left_arm_indexes[1]])

    def get_right_arm_from_joint_position(self, s):
        return np.array(s[self.right_arm_indexes[0] : self.right_arm_indexes[1]])

    def get_left_leg_from_joint_position(self, s):
        return np.array(s[self.left_leg_indexes[0] : self.left_leg_indexes[1]])

    def get_right_leg_from_joint_position(self, s):
        return np.array(s[self.right_leg_indexes[0] : self.right_leg_indexes[1]])

    def get_torso_from_joint_position(self, s):
        return np.array(s[self.torso_indexes[0] : self.torso_indexes[1]])

    def compute_base_pose_left_foot_in_contact(self, s):
        w_H_torso = self.forward_kinematics_fun(self.base_link)
        w_H_leftFoot = self.forward_kinematics_fun(self.left_foot_frame)

        w_H_torso_num = np.array(w_H_torso(np.eye(4), s))
        w_H_lefFoot_num = np.array(w_H_leftFoot(np.eye(4), s))
        w_H_init = np.linalg.inv(w_H_lefFoot_num) @ w_H_torso_num
        return w_H_init

    def get_fitness_function_values(self):
        # For now returns only the CoM height
        base_pose = self.compute_base_pose_left_foot_in_contact(np.zeros(self.NDoF))
        # self.set_state(base_pose, np.zeros(self.NDoF), np.zeros(6), np.zeros(self.NDoF))
        CoM = self.CoM_position_fun()(base_pose, np.zeros(self.NDoF))
        # CoM = CoM_fun()
        return float(CoM[2])

    # def get_mujoco_urdf_string(self):
    #     tempFileOut = tempfile.NamedTemporaryFile(mode = 'w+')
    #     tempFileOut.write(self.urdf_string.decode('utf-8'))
    #     # urdf_path_original = "/home/carlotta/iit_ws/element_hardware-intelligence/Software/OptimizationControlAndHardware/models/urdf/ergoCub/robots/ergoCubSN000/model.urdf"
    #     robot = URDF.load(tempFileOut)
    #     for item in robot.joints: 
    #         if item.name not in (self.joint_name_list): 
    #             item.joint_type = "fixed"
    #     world_joint = Joint("floating_joint","floating", "world", self.base_link)
    #     world_link = Link("world",None, None, None)
    #     robot._links.append(world_link)
    #     robot._joints.append(world_joint)
    #     temp_urdf = tempfile.NamedTemporaryFile(mode = 'w+')
    #     robot.save(temp_urdf)
    #     urdf_string_temp  = temp_urdf.read()
    #     return urdf_string_temp

    # def get_mujoco_model(self): 
    #     # urdf_string = self.get_mujoco_urdf_string()
    #     # urdf_string = urdf_mujoco_file.read()
    #     # mujoco_model = mujoco.MjModel.from_xml_string(urdf_string)
    #     # urdf_path = "/home/carlotta/iit_ws/element_hardware-intelligence/Software/OptimizationControlAndHardware/models/urdf/ergoCub/robots/ergoCubSN000/model_mj.urdf"
    #     urdf_path
    #     mujoco_model = mujoco.MjModel.from_xml_path(self.urdf_path)
    #     path_temp_xml = tempfile.NamedTemporaryFile(mode = 'w+')
    #     mujoco.mj_saveLastXML(path_temp_xml.name,mujoco_model)
        
    #     # Adding the Motors
    #     tree = ET.parse(path_temp_xml)
    #     root = tree.getroot()
    #     mujoco_elem = None
        
    #     for elem in root.iter():
    #         if elem.tag == 'mujoco':
    #             mujoco_elem = elem
    #             break
    #     actuator_entry = ET.Element("actuator")
        
    #     for name_joint in self.joint_name_list:
    #         new_motor_entry = ET.Element('motor')
    #         new_motor_entry.set('name', name_joint)
    #         new_motor_entry.set('joint', name_joint)
    #         new_motor_entry.set('gear', '1') # This can be changed to direclty give motor torques
    #         new_motor_entry.set('ctrlrange', '-100 100') # Also this can be changed to give the exact motor limits
    #         actuator_entry.append(new_motor_entry)
    #     mujoco_elem.append(actuator_entry)

    #     # Adding various assets 
        
    #     asset_entry = ET.Element("asset")
    #     # <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
    #     new_texture_entry=ET.Element('texture')
    #     new_texture_entry.set('type', "skybox")
    #     new_texture_entry.set("builtin","gradient")
    #     new_texture_entry.set("rgb1", ".3 .5 .7")
    #     new_texture_entry.set("rgb2","0 0 0")
    #     new_texture_entry.set("width","32")
    #     new_texture_entry.set("height", "512")
    #     asset_entry.append(new_texture_entry)

    #     # <texture name="body" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    #     new_texture_entry=ET.Element('texture')
    #     new_texture_entry.set('name', "body")
    #     new_texture_entry.set('type', "cube")
    #     new_texture_entry.set("builtin","flat")
    #     new_texture_entry.set("mark", "cross")
    #     new_texture_entry.set("rgb1", "0.8 0.6 0.4")
    #     new_texture_entry.set("rgb2","0.8 0.6 0.4")
    #     new_texture_entry.set("width","127")
    #     new_texture_entry.set("height", "1278")
    #     new_texture_entry.set("markrgb", "1 1 1")
    #     new_texture_entry.set("random","0.01")
    #     asset_entry.append(new_texture_entry)

    #     #<texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    #     new_texture_entry=ET.Element('texture')
    #     new_texture_entry.set('name', "grid")
    #     new_texture_entry.set('type', "2d")
    #     new_texture_entry.set("builtin","checker")
    #     new_texture_entry.set("rgb1", ".1 .2 .3")
    #     new_texture_entry.set("rgb2",".2 .3 .4")
    #     new_texture_entry.set("width","512")
    #     new_texture_entry.set("height", "512")
    #     asset_entry.append(new_texture_entry)

    #     # <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
    #     new_material_entry = ET.Element("material")
    #     new_material_entry.set("name", "body")
    #     new_material_entry.set("texture","body")
    #     new_material_entry.set("texuniform", "true")
    #     new_material_entry.set("rgba", "0.8 0.6 .4 1")
    #     asset_entry.append(new_material_entry)

    #     # <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
    #     new_material_entry = ET.Element("material")
    #     new_material_entry.set("name", "grid")
    #     new_material_entry.set("texture","grid")
    #     new_material_entry.set("texrepeat","1 1")
    #     new_material_entry.set("texuniform", "true")
    #     new_material_entry.set("reflectance", ".2")
    #     asset_entry.append(new_material_entry)

    #     mujoco_elem.append(asset_entry)

    #     ## Adding the floor 
    #     #   <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
    #     world_elem = None
    #     for elem in root.iter():
    #         if elem.tag == 'worldbody':
    #             world_elem = elem
    #             break
    #     floor = ET.Element("geom")
    #     floor.set("name", "floor")
    #     floor.set("size","0 0 .05")
    #     floor.set("type", "plane")
    #     floor.set("material", "grid")
    #     floor.set("condim", "3")
    #     world_elem.append(floor)
    #     new_xml = ET.tostring(tree.getroot(), encoding='unicode')
    #     print(new_xml)
    #     return new_xml
    def get_base_pose_from_contacts(self,s,ds,contact_frames_pose):
        kindyn = self.get_idyntree_kyndyn()
        kindyn.setRobotState(s, ds, self.gravity)

        w_p_b = np.zeros(3)
        w_H_b = np.eye(4)

        for key, value in contact_frames_pose.items():
            w_H_b_i = (
                value
                @ kindyn.getRelativeTransform(key, kindyn.getFloatingBase())
                .asHomogeneousTransform()
                .toNumPy()
            )
            w_p_b = w_p_b + w_H_b_i[0:3, 3]

        w_H_b[0:3, 3] = w_p_b / len(contact_frames_pose)
        # for the time being for the orientation we are just using the orientation of the last contact
        w_H_b[0:3, 0:3] = w_H_b_i[0:3, 0:3]

        return w_H_b
    
    def get_base_pose_from_contacts_mpc(self, s,ds, contact_left, contact_rigth):
        s = iDynTree.VectorDynSize.FromPython(s)
        ds = iDynTree.VectorDynSize.FromPython(ds)
        self.w_b = iDynTree.Twist()
        self.w_b.zero()
        contact_frames_pose = {}
        contact_frame_list = []
        if(contact_left.is_in_contact):
            contact_frames_pose.update({self.left_foot_frame: contact_left.transform.transform()})
            contact_frame_list.append(self.left_foot_frame)
        if(contact_rigth.is_in_contact):
            contact_frames_pose.update({self.right_foot_frame: contact_rigth.transform.transform()})
            contact_frame_list.append(self.right_foot_frame)
        w_H_b=self.get_base_pose_from_contacts(s,ds,contact_frames_pose)
        w_b = self.get_base_velocity_from_contacts(w_H_b, s,ds, contact_frame_list)
        return w_H_b, w_b


    def get_base_velocity_from_contacts(self, H_b, s, ds, contact_frames_list: list):
        kindyn = self.get_idyntree_kyndyn()
        w_b = iDynTree.Twist()
        w_b.zero()
        self.H_b.fromHomogeneousTransform(iDynTree.Matrix4x4.FromPython(H_b))
        kindyn.setRobotState(H_b, s, w_b, ds, self.gravity)
        Jc_multi_contacts = self.get_frames_jacobian(kindyn,contact_frames_list)
        a = Jc_multi_contacts[:, 0:6]
        b = -Jc_multi_contacts[:, 6:]
        w_b = np.linalg.lstsq(
            Jc_multi_contacts[:, 0:6], -Jc_multi_contacts[:, 6:] @ ds.toNumPy(), rcond=-1
        )[0]
        return w_b
    
    def get_frames_jacobian(self,kindyn, frames_list: list):
        Jc_frames = np.zeros([6 * len(frames_list), 6 + self.NDoF])

        for idx, frame_name in enumerate(frames_list):
            Jc = iDynTree.MatrixDynSize(6, 6 + self.NDoF)
            kindyn.getFrameFreeFloatingJacobian(frame_name, Jc)

            Jc_frames[idx * 6 : (idx * 6 + 6), :] = Jc.toNumPy()

        return Jc_frames

    def get_centroidal_momentum_jacobian(self):
        Jcm = iDynTree.MatrixDynSize(6, 6 + self.ndof)
        self.kindyn.getCentroidalTotalMomentumJacobian(Jcm)
        return Jcm.toNumPy()

    def get_centroidal_momentum(self):
        Jcm = self.get_centroidal_momentum_jacobian()
        nu = np.concatenate((self.w_b.toNumPy(), self.ds.toNumPy()))
        return Jcm @ nu