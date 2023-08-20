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