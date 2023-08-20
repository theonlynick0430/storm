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
""" MPC for following specified trajectory

"""
import copy
from isaacgym import gymapi
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
from storm_kit.util_file import get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.differentiable_robot_model.coordinate_transform import matrix_to_quaternion, CoordinateTransform
from storm_kit.mpc.task.traj_task import TrajectoryTask
from obj_centric_stab.utils import *
from isaacgym.torch_utils import quat_conjugate, quat_mul
np.set_printoptions(precision=2)


# helper methods (general rule use torch structs until passing through gym api)

def calc_ee_pose(robot_state, mpc_control, w_T_r, tensor_args):
    state = np.hstack((robot_state['position'], robot_state['velocity'], robot_state['acceleration']))
    state_t = torch.as_tensor(state, **tensor_args).unsqueeze(0)
    pose_state = mpc_control.controller.rollout_fn.get_ee_pose(state_t)
    pos = pose_state['ee_pos_seq'][0]
    quat = pose_state['ee_quat_seq'][0]
    return w_T_r @ pose_from(pos, quat, tensor_args) # in world frame

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

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def linear_action(ee_goal_pose, robot_sim: RobotSim, gym_instance, thresh=0.1, max_steps=500):
    gym = gym_instance.gym
    sim = gym_instance.sim
    ee_goal_pos = ee_goal_pose[:3, 3].unsqueeze(0)
    ee_goal_quat = matrix_to_quaternion(ee_goal_pose[:3, :3].unsqueeze(0))
    step = 0
    while True:
        refresh(gym, sim)
        gym_instance.step()
        step += 1

        w_T_ee = robot_sim.get_ee_pose()
        ee_quat = matrix_to_quaternion(w_T_ee[:3, :3].unsqueeze(0))
        pos_err = ee_goal_pos - w_T_ee[:3, 3].unsqueeze(0)
        orn_err = orientation_error(ee_goal_quat, ee_quat)
        dpose = torch.cat([pos_err, orn_err], 1).t()

        action = robot_sim.control(dpose)
        robot_sim.command_robot(action)

        if torch.linalg.norm(dpose) < thresh:
            return True # success
        if step > max_steps:
            return False # failure

def main(args, gym_instance):
    gym = gym_instance.gym
    sim = gym_instance.sim
    tensor_args = {'device':torch.device("cuda", 0), 'dtype':torch.float32}

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
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    sim_params['collision_model'] = None

    # pose data 
    data_file = args.input
    trajs = list(np.load(data_file, allow_pickle=True).item().values())
    traj = trajs[args.trajectory]
    traj = torch.tensor(traj, **tensor_args)

    # create robot sim: wrapper around isaac sim to add/manage single robot 
    env_handle = gym_instance.env_list[0]
    robot_sim = RobotSim(env_handle=env_handle, gym_instance=gym_instance, **sim_params, tensor_args=tensor_args)
    # spawn robot
    robot_pose = sim_params['robot_pose']
    robot_handle = robot_sim.spawn_robot(robot_pose, coll_id=2)
    
    # get world transform
    # convention x_T_y is transformation from frame x to y
    w_T_r = transform_to_torch(copy.deepcopy(robot_sim.spawn_robot_pose), tensor_args)
    w_robot_coord = CoordinateTransform(trans=w_T_r[0:3,3].unsqueeze(0),
                                        rot=w_T_r[0:3,0:3].unsqueeze(0))
    # create world: wrapper around isaac sim to add/manage objects
    world_instance = World(gym, sim, env_handle, world_params, w_T_r=torch_to_transform(w_T_r), tensor_args=tensor_args)

    # mpc control 
    mpc_control = TrajectoryTask(task_file, robot_file, world_file, tensor_args)

    # spawn ee obj
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002
    cube_color = gymapi.Vec3(51., 153., 255.)/255.
    cube_asset = gym.create_box(sim, 0.05, 0.05, 0.05, asset_options)
    obj_handle = gym.create_actor(env_handle, cube_asset, torch_to_transform(torch.eye(4, **tensor_args)), 'ee_cube', 2, 2, world_instance.ENV_SEG_LABEL)
    gym.set_rigid_body_color(env_handle, obj_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, cube_color)

    # start tensor api (all sim done on gpu)
    gym.prepare_sim(sim)
    robot_sim.init_tensor_api()
    refresh(gym, sim)

    # set transformation for ee obj
    w_T_ee = robot_sim.get_ee_pose()
    ee_T_obj = torch.eye(4, **tensor_args)
    ee_T_obj[:3, 3] = torch.tensor([0., 0., 0.025], **tensor_args)
    obj_pose = w_T_ee @ ee_T_obj
    obj_grasp = torch.inverse(w_T_ee @ ee_T_obj @ torch.inverse(w_T_ee))
    world_instance.set_root_tensor_state(obj_pose, obj_handle)

    # set mpc to traj
    w_t_ee = torch.eye(4, **tensor_args)
    w_t_ee[:3, 3] = w_T_ee[:3, 3]
    traj = w_t_ee @ ee_T_obj @ traj # traj now in world frame
    ee_goal_pose_traj = torch.inverse(w_T_r) @ obj_grasp @ traj # traj for ee given grasp
    mpc_control.update_params(ee_goal_pos_traj=ee_goal_pose_traj[:, :3, 3], 
                              ee_goal_rot_traj=ee_goal_pose_traj[:, :3, :3])
    
    t_step = gym_instance.get_sim_time()
    
    # navigate robot to starting pose
    linear_action(ee_goal_pose_traj[0], robot_sim, gym_instance)

    # main control loop
    sim_dt = mpc_control.exp_params['control_dt']
    real_dt = args.dt
    traj_t_step = 0
    traj_freq = int(real_dt/sim_dt)
    counter: int = 0
    while not gym.query_viewer_has_closed(gym_instance.viewer):
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
            obj_pose = w_T_ee @ ee_T_obj

            # update ee object
            world_instance.set_root_tensor_state(obj_pose, obj_handle)
        
            # run mpc to find optimal action
            command = mpc_control.get_command(t_step, robot_state, control_dt=sim_dt, WAIT=True)
            q_des = copy.deepcopy(command['position'])
            # qd_des = copy.deepcopy(command['velocity'])
            # qdd_des = copy.deepcopy(command['acceleration'])

            # draw top trajs
            gym_instance.clear_lines()
            draw_mpc_top_trajs(gym_instance, mpc_control, w_robot_coord)

            # execute action
            # mpc does not natively support effort control so we must do conversion for now
            # TODO: effort support for mpc
            if robot_sim.controller == "osc":
                ee_goal_pose = calc_ee_pose(command, mpc_control, w_T_r, tensor_args)
                ee_goal_pos = ee_goal_pose[:3, 3].unsqueeze(0)
                ee_goal_quat = matrix_to_quaternion(ee_goal_pose[:3, :3].unsqueeze(0))
                ee_quat = matrix_to_quaternion(w_T_ee[:3, :3].unsqueeze(0))
                pos_err = ee_goal_pos - w_T_ee[:3, 3].unsqueeze(0)
                orn_err = orientation_error(ee_goal_quat, ee_quat)
                dpose = torch.cat([pos_err, orn_err], 1).t()
                action = robot_sim.control(dpose)
            else: # ik
                action = torch.tensor(q_des, **tensor_args)
            robot_sim.command_robot(action)

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
    parser.add_argument('-c', '--controller', type=str, default='osc')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    main(args, gym_instance)
    
