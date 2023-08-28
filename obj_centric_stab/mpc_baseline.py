#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" MPC reference tracking 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib
matplotlib.use('tkagg')
import yaml
import argparse
import numpy as np
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml, get_assets_path, get_mpc_configs_path
from storm_kit.differentiable_robot_model.coordinate_transform import CoordinateTransform
from storm_kit.mpc.task.traj_task import TrajectoryTask
from storm_kit.gym.utils import *
np.set_printoptions(precision=2)


# helper methods (general rule use torch structs until passing through gym api)

def draw_mpc_top_trajs(gym_instance, mpc_control, w_robot_coord: CoordinateTransform):
    top_trajs = mpc_control.top_trajs.cpu().float()
    n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
    w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)
    top_trajs = w_pts.cpu().numpy()
    color = np.array([0.0, 1.0, 0.0])
    for k in range(top_trajs.shape[0]):
        pts = top_trajs[k,:,:]
        color[0] = float(k) / float(top_trajs.shape[0])
        color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
        gym_instance.draw_lines(pts, color=color)

def refresh(gym, sim):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

def linear_action(ee_goal_pose, robot_sim: RobotSim, gym_instance, thresh=0.001, max_steps=500):
    """
    Uses control suite to navigate robot linearly from current to target pose.

    Args:
    ee_goal_pose (1 x 4 x 4, torch.tensor): ee goal pose in world frame 

    Returns:
    success (bool)
    """
    gym = gym_instance.gym
    sim = gym_instance.sim
    ee_goal_pos = ee_goal_pose[:, :3, 3]
    ee_goal_quat = mat2quat(ee_goal_pose[:, :3, :3])
    step = 0
    while True:
        refresh(gym, sim)
        gym_instance.step()
        step += 1

        w_T_ee = robot_sim.get_ee_pose()
        ee_quat = mat2quat(w_T_ee[:, :3, :3])
        pos_err = ee_goal_pos - w_T_ee[:, :3, 3]
        orn_err = orientation_error(ee_goal_quat, ee_quat)
        dpose = torch.cat([pos_err, orn_err], 1).t()

        action = robot_sim.control(dpose)
        robot_sim.command_robot(action)

        if torch.linalg.norm(dpose) < thresh:
            return True # success
        if step > max_steps:
            return False # failure

def main(args, gym_instance):
    # configs
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'
    world_yml = join_path(get_gym_configs_path(), world_file)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)
    robot_yml = join_path(get_gym_configs_path(), robot_file)
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    task_yml = join_path(get_mpc_configs_path(), task_file)
    with open(task_yml) as file:
        task_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    sim_params['collision_model'] = None

    gym = gym_instance.gym
    sim = gym_instance.sim
    tensor_args = {'device':torch.device("cuda", 0), 'dtype':torch.float32}
    horizon = task_params['mppi']['horizon']

    # pose data 
    data_file = args.input
    trajs = list(np.load(data_file, allow_pickle=True).item().values())
    traj = trajs[args.trajectory]
    traj = torch.tensor(traj, **tensor_args)

    # create robot sim: wrapper around isaac sim to add/manage single robot 
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, tensor_args=tensor_args)
    # spawn robot
    robot_pose = sim_params['robot_pose']
    env_handle = gym_instance.env_list[0]
    robot_handle = robot_sim.spawn_robot(env_handle, robot_pose, coll_id=2)

    # convention x_T_y is transformation from frame y to x
    
    # get world transform
    w_T_r = transform_to_torch(copy.deepcopy(robot_sim.spawn_robot_pose), tensor_args)
    w_robot_coord = CoordinateTransform(trans=w_T_r[:, :3, 3],
                                        rot=w_T_r[:, :3, :3])
    # create world: wrapper around isaac sim to add/manage objects
    world_instance = World(gym, sim, env_handle, world_params, w_T_r=torch_to_transform(w_T_r))

    # spawn ee object
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002
    cube_color = gymapi.Vec3(51., 153., 255.)/255.
    cube_asset = gym.create_box(sim, 0.05, 0.05, 0.05, asset_options)
    cube_handle = gym.create_actor(env_handle, cube_asset, torch_to_transform(torch.eye(4, **tensor_args).unsqueeze(0)), 'ee_cube', 2, 2, world_instance.ENV_SEG_LABEL)
    gym.set_rigid_body_color(env_handle, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cube_color)

    # start tensor api (all sim done on gpu)
    gym.prepare_sim(sim)
    robot_sim.init_tensor_api()
    refresh(gym, sim)

    w_T_ee = robot_sim.get_ee_pose()
    r_T_ee = torch.inverse(w_T_r) @ w_T_ee
    ee_T_obj = torch.eye(4, **tensor_args)
    ee_T_obj[:3, 3] = torch.tensor([0., 0., 0.025], **tensor_args)
    # obj_grasp converts pose of ee in robot frame to pose of obj in robot frame
    obj_grasp = r_T_ee @ ee_T_obj @ torch.inverse(r_T_ee)
    # initial pose of ee in world frame
    init_pose = copy.deepcopy(w_T_ee)
    init_pose[0, :3, :3] = torch.eye(3, **tensor_args)
    obj_goal_pose_traj = torch.inverse(w_T_r) @ init_pose @ ee_T_obj @ traj # in robot frame

    # mpc control 
    mpc_control = TrajectoryTask(obj_grasp, task_file, robot_file, world_file, tensor_args)
    # send traj to mpc
    mpc_control.update_params(obj_goal_pos_traj=obj_goal_pose_traj[:, :3, 3], 
                              obj_goal_rot_traj=obj_goal_pose_traj[:, :3, :3])
    
    t_step = gym_instance.get_sim_time()
    linear_action(w_T_r @ torch.inverse(obj_grasp) @ obj_goal_pose_traj[0], robot_sim, gym_instance)
    
    # main control loop
    sim_dt = mpc_control.exp_params['control_dt']
    real_dt = args.dt
    # t_step = gym_instance.get_sim_time()
    traj_t_step = 0
    traj_freq = int(real_dt/sim_dt)
    counter: int = 0
    while t_step >= 0:
        try:
            gym_instance.step()
            refresh(gym, sim)
            t_step += sim_dt
            counter += 1

            if counter == traj_freq:
                counter = 0
                traj_t_step += 1
                print(traj_t_step)
                traj_t_step = min(traj_t_step, traj.shape[0]-1)
                mpc_control.update_params(active_idx=traj_t_step)
            
            robot_state = copy.deepcopy(robot_sim.get_state())
            w_T_ee = robot_sim.get_ee_pose()
            cube_pose = w_T_ee @ ee_T_obj

            # update ee obj pose
            world_instance.set_root_tensor_state(cube_pose, cube_handle)
        
            # run mpc to find optimal action
            command = mpc_control.get_command(t_step, robot_state, control_dt=sim_dt, WAIT=True)
            q_des = copy.deepcopy(command['position'])

            # qd_des = copy.deepcopy(command['velocity'])
            # qdd_des = copy.deepcopy(command['acceleration'])
            # draw top trajs
            gym_instance.clear_lines()
            draw_mpc_top_trajs(gym_instance, mpc_control, w_robot_coord)
            # execute action
            robot_sim.command_robot(torch.tensor(q_des, **tensor_args))            

            # ee_error = mpc_control.get_current_error(robot_state)
            # print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt),
            #       "{:.3f}".format(mpc_control.mpc_dt))
            
        except KeyboardInterrupt:
            print('Finished sim session...closing')
            break

    mpc_control.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('-i', '--input', type=str, required=True, help='Pose data file')
    parser.add_argument('-t', '--trajectory', type=int, default=0, help='Index of trajectory to follow')
    parser.add_argument('-dt', type=float, default=0.5, help='Time between each pose data. Must be multiple of control_dt')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    main(args, gym_instance)
    
