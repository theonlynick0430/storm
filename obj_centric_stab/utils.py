import copy
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path, get_mpc_configs_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, matrix_to_quaternion, CoordinateTransform
from storm_kit.mpc.task.traj_task import TrajectoryTask
from scipy.spatial.transform import Rotation as R
from isaacgym.torch_utils import quat_conjugate, quat_mul, get_euler_xyz
np.set_printoptions(precision=2)


def transform_to_torch(transform, tensor_args):
    mat = torch.eye(4, **tensor_args)
    quat = torch.tensor([transform.r.w, transform.r.x, transform.r.y, transform.r.z], **tensor_args).unsqueeze(0)
    rot_mat = quaternion_to_matrix(quat).squeeze(0)
    mat[0,3] = transform.p.x
    mat[1,3] = transform.p.y
    mat[2,3] = transform.p.z
    mat[:3,:3] = rot_mat
    return mat

def torch_to_transform(mat):
    vec3 = torch_to_vec3(mat[:3, 3])
    quat = torch_to_quat(matrix_to_quaternion(mat[:3, :3].unsqueeze(0)).squeeze(0))
    return gymapi.Transform(vec3, quat)

def quat_to_torch(quat, tensor_args):
    return torch.tensor([quat.w, quat.x, quat.y, quat.z], **tensor_args) # assume wxyz format

def torch_to_quat(arr):
    arr_cpu = arr.cpu()
    return gymapi.Quat(arr_cpu[1], arr_cpu[2], arr_cpu[3], arr_cpu[0]) # xyzw format

def vec3_to_torch(vec3, tensor_args):
    return torch.tensor([vec3.x, vec3.y, vec3.z], **tensor_args)

def torch_to_vec3(arr):
    arr_cpu = arr.cpu()
    return gymapi.Vec3(arr_cpu[0], arr_cpu[1], arr_cpu[2])

def pose_from(pos, quat, tensor_args):
    rot_mat = quaternion_to_matrix(quat.unsqueeze(0)).squeeze(0)
    pose = torch.eye(4, **tensor_args)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = pos
    return pose

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

# dpose should be the goal_state-current_state in world frame
def control_ik(dpose, j_eef, tensor_args, damping = 0.05):
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, **tensor_args) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(1, 7)
    return u

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# dpose should be the goal_state-current_state in world frame
def control_osc(dpose, j_eef, default_dof_pos_tensor, mm, dof_pos, dof_vel, ee_vel, tensor_args):
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * ee_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=tensor_args['device']).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)