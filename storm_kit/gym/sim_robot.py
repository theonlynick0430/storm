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

# This file contains a generic robot class that can load a robot asset into sim and gives access to robot's state and control.


import copy

import numpy as np
from quaternion import from_rotation_matrix, as_float_array, as_rotation_matrix, as_quat_array
try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
    from isaacgym import gymtorch
except Exception:
    print("ERROR: gym not loaded, this is okay when generating doc")

import torch
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, matrix_to_quaternion, CoordinateTransform
from .helpers import load_struct_from_dict
from ..util_file import join_path

def pose_from(pos, quat, tensor_args):
    rot_mat = quaternion_to_matrix(quat.unsqueeze(0))[0]
    pose = torch.eye(4, **tensor_args)
    pose[:3, :3] = rot_mat
    pose[:3, 3] = pos
    return pose

def inv_transform(gym_transform):
    mat = np.eye(4)
    mat[0:3, 3] = np.ravel([gym_transform.p.x,gym_transform.p.y, gym_transform.p.z])
    # get rotation matrix from quat:
    q = gym_transform.r

    rot = as_rotation_matrix(as_quat_array([q.w, q.x, q.y, q.z]))
    mat[0:3, 0:3] = rot
    inv_mat = np.linalg.inv(mat)

    quat = as_float_array(from_rotation_matrix(inv_mat[0:3,0:3]))
    new_transform = gymapi.Transform(p=gymapi.Vec3(inv_mat[0,3], inv_mat[1,3],
                                                   inv_mat[2,3]),
                                     r=gymapi.Quat(quat[1],
                                                   quat[2],
                                                   quat[3],
                                                   quat[0]))
    
    return new_transform
# Write some helper functions:
def pose_from_gym(gym_pose):
    pose = np.array([gym_pose.p.x, gym_pose.p.y, gym_pose.p.z,
                     gym_pose.r.x, gym_pose.r.y, gym_pose.r.z, gym_pose.r.w])
    return pose

class RobotSim():
    def __init__(self, device='cpu', gym_instance=None, sim_instance=None,
                 asset_root='', sim_urdf='', asset_options='', init_state=None, collision_model=None, **kwargs):
        self.gym = gym_instance
        self.sim = sim_instance
        self.device = device
        self.dof = None
        self.init_state = init_state
        self.joint_names = []
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options = load_struct_from_dict(robot_asset_options, asset_options)

        self.camera_handle = None
        self.collision_model_params = collision_model
        self.DEPTH_CLIP_RANGE = 6.0
        self.ENV_SEG_LABEL = 1
        self.ROBOT_SEG_LABEL = 2
        
        self.robot_asset = self.load_robot_asset(sim_urdf,
                                                 robot_asset_options,
                                                 asset_root)

        
    def init_sim(self, gym_instance, sim_instance):
        self.gym = gym_instance
        self.sim = sim_instance
        
    def load_robot_asset(self, sim_urdf, asset_options, asset_root):

        if ((self.gym is None) or (self.sim is None)):
            raise AssertionError
        robot_asset = self.gym.load_asset(self.sim, asset_root,
                                          sim_urdf, asset_options)
        #print(asset_options.disable_gravity)
        return robot_asset

    def spawn_robot(self, env_handle, robot_pose, robot_asset=None, coll_id=-1):
        self.env_handle = env_handle
        self.robot_name = "robot"
        self.tensor_args = {'device':torch.device("cuda", 0), 'dtype':torch.float32}
        self.controller = "ik"
        p = gymapi.Vec3(robot_pose[0], robot_pose[1], robot_pose[2])
        robot_pose = gymapi.Transform(p=p, r=gymapi.Quat(robot_pose[3], robot_pose[4], robot_pose[5], robot_pose[6]))
        self.spawn_robot_pose = robot_pose
        
        if(robot_asset is None):
            robot_asset = self.robot_asset
        self.robot_handle = self.gym.create_actor(self.env_handle, robot_asset,
                                             robot_pose, self.robot_name, coll_id, coll_id, self.ROBOT_SEG_LABEL) # coll_id, mask_filter, mask_vision

        self.shape_props = self.gym.get_actor_rigid_shape_properties(self.env_handle, self.robot_handle)
        for i in range(len(self.shape_props)):
            self.shape_props[i].friction = 1.5
        self.gym.set_actor_rigid_shape_properties(self.env_handle, self.robot_handle, self.shape_props)
        
        self.joint_names = self.gym.get_actor_dof_names(self.env_handle, self.robot_handle)
        self.sim_dof_count = self.gym.get_sim_dof_count(self.sim)
        self.dof = self.gym.get_actor_dof_count(self.env_handle, self.robot_handle)
        self.dof_idx = self.gym.get_actor_dof_index(self.env_handle, self.robot_handle, 0, gymapi.DOMAIN_SIM)
        self.rb_count = self.gym.get_actor_rigid_body_count(self.env_handle, self.robot_handle)
        self.rb_idx = self.gym.get_actor_rigid_body_index(self.env_handle, self.robot_handle, 0, gymapi.DOMAIN_SIM)
        self.rs_idx = self.gym.get_actor_index(self.env_handle, self.robot_handle, gymapi.DOMAIN_SIM)
        self.actor_indices = gymtorch.unwrap_tensor(torch.arange(self.dof_idx, self.dof_idx+self.dof, 
                                                                 dtype=torch.int32, device=self.tensor_args['device']))
        franka_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)
        self.ee_link_idx = franka_link_dict["ee_link"]

        self.robot_dof_props = self.gym.get_actor_dof_properties(self.env_handle, self.robot_handle)
        self.robot_lower_limits = self.robot_dof_props["lower"]
        self.robot_upper_limits = self.robot_dof_props["upper"]
        if self.controller == "ik":
            self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.robot_dof_props["stiffness"][:7].fill(400.0)
            self.robot_dof_props["damping"][:7].fill(40.0)
        else: # osc
            self.robot_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
            self.robot_dof_props["stiffness"][:7].fill(0.0)
            self.robot_dof_props["damping"][:7].fill(0.0)
        if self.init_state is None:
            robot_mids = 0.5 * (self.robot_upper_limits + self.robot_lower_limits)
            self.init_state = torch.tensor(robot_mids, **self.tensor_args).cpu().numpy()
        self.gym.set_actor_dof_properties(self.env_handle, self.robot_handle, self.robot_dof_props)  
        self.gym.set_actor_dof_position_targets(self.env_handle, self.robot_handle, self.init_state)   
        
        robot_dof_states = copy.deepcopy(self.gym.get_actor_dof_states(self.env_handle, self.robot_handle, gymapi.STATE_ALL))
        for i in range(self.dof):
            robot_dof_states['pos'][i] = self.init_state[i]
            robot_dof_states['vel'][i] = 0.0
        self.init_robot_state = robot_dof_states
        self.gym.set_actor_dof_states(self.env_handle, self.robot_handle, self.init_robot_state, gymapi.STATE_ALL)

        if(self.collision_model_params is not None):
            self.init_collision_model(self.collision_model_params, self.env_handle, self.robot_handle)

        return self.robot_handle
    
    def init_tensor_api(self):
        # all tensors will automatically update on refresh 
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "robot")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        print("EE LINK IDX")
        print(self.ee_link_idx)
        self.j_eef = jacobian[:, self.ee_link_idx-1, :, :7]

        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "robot")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7] 

        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_pos = self.dof_states[0:7, 0]
        self.dof_vel = self.dof_states[0:7, 1]

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        # in world frame
        self.ee_pos = self.rb_states[self.rb_idx+self.rb_count-1, :3]
        self.ee_quat = self.rb_states[self.rb_idx+self.rb_count-1, 3:7]
        self.ee_vel = self.rb_states[self.rb_idx+self.rb_count-1, 7:]

        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_tensor = gymtorch.wrap_tensor(_root_tensor)
        self.rs_pos = self.root_tensor[self.rs_idx, 0:3]
        self.rs_quat = self.root_tensor[self.rs_idx, 3:7]
        self.rs_vel = self.root_tensor[self.rs_idx, 7:]

    def get_state(self, env_handle, robot_handle):
        robot_state = self.gym.get_actor_dof_states(env_handle, robot_handle, gymapi.STATE_ALL)
        
        # reformat state to be similar ros jointstate:
        joint_state = {'name':self.joint_names, 'position':[], 'velocity':[], 'acceleration':[]}

        for i in range(len(robot_state)):
            joint_state['position'].append(robot_state[i][0])
            joint_state['velocity'].append(robot_state[i][1])
        joint_state['position'] = np.ravel(joint_state['position'])
        joint_state['velocity'] = np.ravel(joint_state['velocity'])
        joint_state['acceleration'] = np.ravel(joint_state['velocity'])*0.0
        
        return joint_state
    
    def get_state2(self):
        # _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        # self.dof_states = gymtorch.wrap_tensor(_dof_states)
        # self.dof_pos = self.dof_states[0:7, 0]
        # self.dof_vel = self.dof_states[0:7, 1]
        # reformat state to be similar ros jointstate:
        joint_state = {'name':self.joint_names, 'position':[], 'velocity':[], 'acceleration':[]}
        joint_state['position'] = self.dof_pos.cpu().numpy()
        joint_state['velocity'] = self.dof_vel.cpu().numpy()
        joint_state['acceleration'] = np.zeros_like(joint_state['velocity'])
        return joint_state
    
    def get_ee_pose(self):
        print("GETTING EE POSE")
        duh = copy.deepcopy(self.ee_quat)
        w = self.ee_quat[-1]
        duh[1:] = self.ee_quat[:3]
        duh[0] = w
        duh2 = copy.deepcopy(self.rs_quat)
        w2 = self.rs_quat[-1]
        duh2[1:] = self.rs_quat[:3]
        duh2[0] = w2
        print(self.rs_pos)
        print(self.rs_quat)
        print(duh2)
        print(self.ee_pos)
        print(self.ee_quat)
        print(duh)
        return pose_from(copy.deepcopy(self.ee_pos), duh, self.tensor_args)

    def command_robot(self, tau, env_handle, robot_handle):
        self.gym.apply_actor_dof_efforts(env_handle, robot_handle, np.float32(tau))
        
    def command_robot_position(self, q_des, env_handle, robot_handle):
        self.gym.set_actor_dof_position_targets(env_handle, robot_handle, np.float32(q_des))

    def command_robot_position2(self, pos):
        # _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        # self.dof_states = gymtorch.wrap_tensor(_dof_states)
        # self.dof_pos = self.dof_states[0:7, 0]
        # self.dof_vel = self.dof_states[0:7, 1]
        pos_action = copy.deepcopy(self.dof_states)[:, 0]
        pos_action[0:7] = pos
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_action.contiguous()))

    def control(self, dpose):
        if self.controller == "osc":
            return self.control_osc(dpose).squeeze(0)
        else: # ik
            return self.dof_pos + self.control_ik(dpose).squeeze(0)

    def control_ik(self, dpose):
        print("CONTROL")
        print(dpose)
        print(self.j_eef)
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        lmbda = torch.eye(6, **self.tensor_args) * (0.05 ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(1, 7)
        return u

    def set_robot_state(self, q_des, qd_des, env_handle, robot_handle):
        robot_dof_states = copy.deepcopy(self.gym.get_actor_dof_states(env_handle, robot_handle,
                                                                       gymapi.STATE_ALL))

        for i in range(len(robot_dof_states['pos'])):
            robot_dof_states['pos'][i] = q_des[i]
            robot_dof_states['vel'][i] = qd_des[i]
        self.init_robot_state = robot_dof_states
        self.gym.set_actor_dof_states(env_handle, robot_handle, robot_dof_states, gymapi.STATE_ALL)

    def update_collision_model(self, link_poses, env_ptr, robot_handle):
        w_T_r = self.spawn_robot_pose
        for i in range(len(link_poses)):
            #print(i)
            link = self.link_colls[i]
            link_pose = gymapi.Transform()
            link_pose.p = gymapi.Vec3(link_poses[i][0], link_poses[i][1], link_poses[i][2])
            link_pose.r = gymapi.Quat(link_poses[i][4], link_poses[i][5], link_poses[i][6],link_poses[i][3])
            w_p1 = w_T_r * link_pose * link['pose_offset'] * link['base']
            self.gym.set_rigid_transform(env_ptr, link['p1_body_handle'], w_p1)
            w_p2 = w_T_r * link_pose * link['pose_offset'] * link['tip']
            self.gym.set_rigid_transform(env_ptr, link['p2_body_handle'], w_p2)

    def init_collision_model(self, robot_collision_params, env_ptr, robot_handle):
        
        # get robot w_T_r
        w_T_r = self.spawn_robot_pose
        # transform all points based on this:
        
        robot_links = robot_collision_params['link_objs']
        #x,y,z = 0.1, 0.1, 0.2
        obs_color = gymapi.Vec3(0.0, 0.5, 1.0)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        # link pose is in robot base frame:
        link_pose = gymapi.Transform()
        link_pose.p = gymapi.Vec3(0, 0, 0)
        link_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.link_colls = []
        
        for j in robot_links:
            #print(j)
            pose = robot_links[j]['pose_offset']
            pose_offset = gymapi.Transform()
            pose_offset.p = gymapi.Vec3(pose[0], pose[1], pose[2])
            #pose_offset.r = gymapi.Quat.from_rpy(pose[3], pose[4], pose[5])
            r = robot_links[j]['radius']
            base = np.ravel(robot_links[j]['base'])
            tip = np.ravel(robot_links[j]['tip'])
            width = np.linalg.norm(base - tip)
            pt1_pose = gymapi.Transform()
            pt1_pose.p = gymapi.Vec3(base[0], base[1], base[2])
            link_p1_asset = self.gym.create_sphere(self.sim, r, asset_options)
            link_p1_handle = self.gym.create_actor(env_ptr, link_p1_asset,w_T_r * pose_offset * pt1_pose, j, 2, 2)
            self.gym.set_rigid_body_color(env_ptr, link_p1_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          obs_color)
            link_p1_body = self.gym.get_actor_rigid_body_handle(env_ptr, link_p1_handle, 0)
            
            pt2_pose = gymapi.Transform()
            pt2_pose.p = gymapi.Vec3(tip[0], tip[1], tip[2])
            link_p2_asset = self.gym.create_sphere(self.sim, r, asset_options)
            link_p2_handle = self.gym.create_actor(env_ptr, link_p2_asset, w_T_r * pose_offset * pt2_pose, j, 2, 2)
            link_p2_body = self.gym.get_actor_rigid_body_handle(env_ptr, link_p2_handle, 0)
            self.gym.set_rigid_body_color(env_ptr, link_p2_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION,
                                          obs_color)
        
    
            link_coll = {'base':pt1_pose, 'tip':pt2_pose, 'pose_offset':pose_offset, 'radius':r,
                         'p1_body_handle':link_p1_body, 'p2_body_handle': link_p2_body}
            self.link_colls.append(link_coll)

    def spawn_camera(self, env_ptr, fov, width, height, robot_camera_pose):
        """
        Spawn a camera in the environment
        Args:
        env_ptr: environment pointer
        fov, width, height: camera params
        robot_camera_pose: Camera pose w.r.t robot_body_handle [x, y, z, qx, qy, qz, qw]
        """
        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = fov
        camera_props.height = height
        camera_props.width = width
        camera_props.use_collision_geometry = False

        self.num_cameras = 1
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        robot_camera_pose = gymapi.Transform(
            gymapi.Vec3(robot_camera_pose[0], robot_camera_pose[1], robot_camera_pose[2]),
            gymapi.Quat(robot_camera_pose[3], robot_camera_pose[4], robot_camera_pose[5], robot_camera_pose[6]))

        # quat (q.x, q.y, q.z, q.w)
        # as_float_array(q.w, q.x, q.y, q.z)
        world_camera_pose = self.spawn_robot_pose * robot_camera_pose
        
        #print('Spawn camera pose:',world_camera_pose.p)
        self.gym.set_camera_transform(
            camera_handle,
            env_ptr,
            world_camera_pose)

        self.camera_handle = camera_handle
        
        return camera_handle

        
        
    def observe_camera(self, env_ptr):
        self.gym.render_all_camera_sensors(self.sim)
        self.current_env_observations = []
        
        camera_handle = self.camera_handle

        w_c_mat = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle).T
        #print('View matrix',w_c_mat)
        #p = gymapi.Vec3(w_c_mat[3,0], w_c_mat[3,1], w_c_mat[3,2])
        #p = gymapi.Vec3(w_c_mat[0,3], w_c_mat[1,3], w_c_mat[2,3])
        #quat = as_float_array(from_rotation_matrix(w_c_mat[0:3, 0:3]))
        #r = gymapi.Quat(quat[1], quat[2], quat[3], quat[0])
        camera_pose = self.spawn_robot_pose.inverse()

        proj_matrix = self.gym.get_camera_proj_matrix(
            self.sim, env_ptr, camera_handle
        )
        view_matrix = self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)#.T
        #view_matrix = view_matrix_t
        #view_matrix[0:3,3] = view_matrix_t[3,0:3]
        #view_matrix[3,0:3] = 0.0
        q = camera_pose.r
        p = camera_pose.p
        camera_pose = [p.x,p.y, p.z, q.x, q.y, q.z, q.w]
        
        
        color_image = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_COLOR)
        color_image = np.reshape(color_image, [480, 640, 4])[:, :, :3]

        depth_image = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_DEPTH,
        )
        depth_image[depth_image == np.inf] = 0
        #depth_image[depth_image > self.DEPTH_CLIP_RANGE] = 0
        segmentation = self.gym.get_camera_image(
            self.sim,
            env_ptr,
            camera_handle,
            gymapi.IMAGE_SEGMENTATION,
        )
        
        camera_data = {'color':color_image, 'depth':depth_image,
                       'segmentation':segmentation, 'robot_camera_pose':camera_pose,
                       'proj_matrix':proj_matrix, 'label_map':{'robot': self.ROBOT_SEG_LABEL,
                                                               'ground': 0},
                       'view_matrix':view_matrix,
                       'world_robot_pose': self.spawn_robot_pose}
        return camera_data
