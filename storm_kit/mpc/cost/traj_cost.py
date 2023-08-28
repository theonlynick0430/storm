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
import torch
import torch.nn as nn
# import torch.nn.functional as F

from .gaussian_projection import GaussianProjection
from ...differentiable_robot_model.coordinate_transform import matrix_to_euler_angles
class TrajectoryCost(nn.Module):

    def __init__(self, obj_grasp, weight, vec_weight=[], position_gaussian_params={}, orientation_gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, hinge_val=100.0,
                 convergence_val=[0.0,0.0]):
        super(TrajectoryCost, self).__init__()
        self.obj_grasp = obj_grasp
        self.tensor_args = tensor_args
        self.weight = weight
        self.vec_weight = torch.as_tensor(vec_weight, **tensor_args)
        self.ori_weight = self.vec_weight[0:3]
        self.pos_weight = self.vec_weight[3:6]

        self.px = torch.tensor([1.0,0.0,0.0], **self.tensor_args).T
        self.py = torch.tensor([0.0,1.0,0.0], **self.tensor_args).T
        self.pz = torch.tensor([0.0,0.0,1.0], **self.tensor_args).T
        
        self.I = torch.eye(3, 3,**self.tensor_args)
        self.Z = torch.zeros(1,**self.tensor_args)

        self.position_gaussian = GaussianProjection(gaussian_params=position_gaussian_params)
        self.orientation_gaussian = GaussianProjection(gaussian_params=orientation_gaussian_params)
        self.hinge_val = hinge_val
        self.convergence_val = convergence_val
        self.dtype = self.tensor_args['dtype']
        self.device = self.tensor_args['device']

    def forward(self, ee_pos_batch, ee_rot_batch, obj_goal_pos_traj, obj_goal_rot_traj):     
        """
        Computes cost between sampled and goal trajectories using l2 distance.

        Args: 
        ee_pos_batch (batch_size x horizon x 3, torch.tensor): sampled batch of ee pos in robot frame
        ee_rot_batch (batch_size x horizon x 3 x 3, torch.tensor): sampled batch of ee rot in robot frame
        obj_goal_pos_traj (Y x 3, torch.tensor): desired traj of obj pos in robot frame
        obj_goal_rot_traj (Y x 3 x 3, torch.tensor): desired traj of obj rot in robot frame

        Returns: 
        cost (batch_size x horizon, torch.tensor): cost matrix that enumerates batch_size and horizon
        """   
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_rot_batch = ee_rot_batch.to(device=self.device,
                                       dtype=self.dtype)
        obj_goal_pos_traj = obj_goal_pos_traj.to(device=self.device,
                                     dtype=self.dtype)
        obj_goal_rot_traj = obj_goal_rot_traj.to(device=self.device,
                                     dtype=self.dtype)
        
        batch_size = ee_pos_batch.shape[0]
        horizon = ee_pos_batch.shape[1]
        
        # make batch and goal horizon match
        if obj_goal_pos_traj.shape[0] > horizon:
            obj_goal_pos_traj = obj_goal_pos_traj[:horizon]
            obj_goal_rot_traj = obj_goal_rot_traj[:horizon]
        elif obj_goal_pos_traj.shape[0] < horizon:
            last_val_pos = obj_goal_pos_traj[-1:]
            last_val_rot = obj_goal_rot_traj[-1:]
            repeats = horizon - obj_goal_pos_traj.shape[0]
            obj_goal_pos_traj = torch.cat([obj_goal_pos_traj, last_val_pos.repeat(repeats, 1)])
            obj_goal_rot_traj = torch.cat([obj_goal_rot_traj, last_val_rot.repeat(repeats, 1, 1)])

        # convert ee pose into obj pose
        obj_pos_batch = self.obj_grasp[:, :3, :3] @ ee_pos_batch.view(batch_size, horizon, 3, 1) + self.obj_grasp[:, :3, 3:4]
        obj_rot_batch = self.obj_grasp[:, :3, :3] @ ee_rot_batch

        # compute error in obj frame
        obj_goal_rot_inv_traj = obj_goal_rot_traj.transpose(-1, -2)
        goal2ee_trans = obj_goal_rot_inv_traj @ (obj_pos_batch - obj_goal_pos_traj.view(horizon, 3, 1))
        goal2ee_rot = obj_goal_rot_inv_traj @ obj_rot_batch
        goal2ee_eul = matrix_to_euler_angles(goal2ee_rot)
        pos_err = torch.norm(self.pos_weight * goal2ee_trans.squeeze(-1), dim=-1)
        ori_err = torch.norm(self.ori_weight * goal2ee_eul, dim=-1)
        ori_err[ori_err < self.convergence_val[0]] = 0.0
        pos_err[pos_err < self.convergence_val[1]] = 0.0
        cost = self.weight[0] * self.orientation_gaussian(ori_err) + self.weight[1] * self.position_gaussian(pos_err)

        return cost.to(inp_device), ori_err, pos_err


