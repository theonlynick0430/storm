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

class TrajectoryCost(nn.Module):

    def __init__(self, weight, vec_weight=[], position_gaussian_params={}, orientation_gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, hinge_val=100.0,
                 convergence_val=[0.0,0.0]):
        super(TrajectoryCost, self).__init__()
        self.tensor_args = tensor_args
        self.I = torch.eye(3, 3, **tensor_args)
        self.weight = weight
        self.vec_weight = torch.as_tensor(vec_weight, **tensor_args)
        self.rot_weight = self.vec_weight[0:3]
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

    # arg shapes
    # ee_pos_batch: (batch_size, x, 3)
    # ee_rot_batch: (batch_size, x, 3, 3)
    # ee_goal_pos_traj: (y, 3)
    # ee_goal_rot_traj: (y, 3, 3))
    def forward(self, ee_pos_batch, ee_rot_batch, ee_goal_pos_traj, ee_goal_rot_traj):        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_rot_batch = ee_rot_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_goal_pos_traj = ee_goal_pos_traj.to(device=self.device,
                                     dtype=self.dtype)
        ee_goal_rot_traj = ee_goal_rot_traj.to(device=self.device,
                                     dtype=self.dtype)
        
        batch_size = ee_pos_batch.shape[0]
        horizon = ee_pos_batch.shape[1]

        if ee_goal_pos_traj.shape[0] > horizon:
            ee_goal_pos_traj = ee_goal_pos_traj[:horizon]
            ee_goal_rot_traj = ee_goal_rot_traj[:horizon]
        elif ee_goal_pos_traj.shape[0] < horizon:
            last_val_pos = ee_goal_pos_traj[-1:]
            last_val_rot = ee_goal_rot_traj[-1:]
            repeats = horizon - ee_goal_pos_traj.shape[0]
            ee_goal_pos_traj = torch.cat([ee_goal_pos_traj, last_val_pos.repeat(repeats, 1)])
            ee_goal_rot_traj = torch.cat([ee_goal_rot_traj, last_val_rot.repeat(repeats, 1, 1)])

        cost = torch.zeros((batch_size, horizon), **self.tensor_args)
        goal_dist = torch.zeros((batch_size, horizon), **self.tensor_args)
        pos_err = torch.zeros((batch_size, horizon), **self.tensor_args)
        rot_err_norm = torch.zeros((batch_size, horizon), **self.tensor_args)
        rot_err = torch.zeros((batch_size, horizon), **self.tensor_args)
        
        for i in range(batch_size):
            ee_pos_traj = ee_pos_batch[i]
            ee_rot_traj = ee_rot_batch[i]

            # generate necessary mats
            ee_rot_diag = torch.block_diag(*ee_rot_traj)
            ee_goal_rot_diag = torch.block_diag(*ee_goal_rot_traj)
            ee_pos_col = ee_pos_traj.reshape(-1, 1)
            ee_goal_pos_col = ee_goal_pos_traj.reshape(-1, 1)
            ee_rot_col = ee_rot_traj.reshape(-1, 3)

            # convention: x_T_y -> transformation from frame y to x

            # all transformation in robot frame
            w_R_g = ee_goal_rot_diag.t() # inverse
            ee_R_g = w_R_g @ ee_rot_col
            o_t_g = -1.0 * w_R_g @ ee_goal_pos_col
            w_T_ee = ee_rot_diag.t() # inverse 
            ee_t_o = w_T_ee @ ee_pos_col
            ee_t_g = o_t_g + ee_t_o
        
            # reshape after mat mult is done
            ee_R_g = ee_R_g.reshape(-1, 3, 3)
            ee_t_g = ee_t_g.reshape(-1, 3)

            goal_dist[i, :] = torch.norm(self.pos_weight * ee_t_g, p=2, dim=-1)
            pos_err[i, :] = (torch.sum(torch.square(self.pos_weight * ee_t_g), dim=-1))
            _rot_err = self.I - ee_R_g
            _rot_err = torch.norm(_rot_err, dim=-1)
            rot_err_norm[i, :] = torch.norm(self.rot_weight * _rot_err, p=2, dim=-1)
            rot_err[i, :] = torch.square(torch.sum(self.rot_weight * _rot_err, dim=-1))

        if(self.hinge_val > 0.0):
            rot_err = torch.where(goal_dist <= self.hinge_val, rot_err, self.Z) #hard hinge

        rot_err[rot_err < self.convergence_val[0]] = 0.0
        pos_err[pos_err < self.convergence_val[1]] = 0.0
        cost = self.weight[0] * self.orientation_gaussian(torch.sqrt(rot_err)) + self.weight[1] * self.position_gaussian(torch.sqrt(pos_err))

        return cost.to(inp_device), rot_err_norm, goal_dist


