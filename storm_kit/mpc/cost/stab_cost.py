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

class StabilizeCost(nn.Module):

    def __init__(self, obj_grasp, weight, vec_weight=[], position_gaussian_params={}, orientation_gaussian_params={}, tensor_args={'device':"cpu", 'dtype':torch.float32}, hinge_val=100.0,
                 convergence_val=[0.0,0.0]):
        super(StabilizeCost, self).__init__()
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

    # arg shapes
    # ee_pos_batch: (batch_size, x, 3)
    # ee_rot_batch: (batch_size, x, 3, 3)
    # obj_init_pos: (3)
    # obj_init_rot: (3, 3))
    def forward(self, ee_pos_batch, ee_rot_batch, obj_init_pos, obj_init_rot):        
        inp_device = ee_pos_batch.device
        ee_pos_batch = ee_pos_batch.to(device=self.device,
                                       dtype=self.dtype)
        ee_rot_batch = ee_rot_batch.to(device=self.device,
                                       dtype=self.dtype)
        obj_init_pos = obj_init_pos.to(device=self.device,
                                     dtype=self.dtype)
        obj_init_rot = obj_init_rot.to(device=self.device,
                                     dtype=self.dtype)
        
        batch_size = ee_pos_batch.shape[0]
        horizon = ee_pos_batch.shape[1]

        # convert ee pose into obj pose
        obj_pos_batch = self.obj_grasp[:, :3, :3] @ ee_pos_batch.view(batch_size, horizon, 3, 1) + self.obj_grasp[:, :3, 3:4]
        obj_rot_batch = self.obj_grasp[:, :3, :3] @ ee_rot_batch

        # compute error in obj frame
        obj_init_rot_inv = obj_init_rot.transpose(-1, -2)
        trans = obj_init_rot_inv @ (obj_pos_batch - obj_init_pos.view(3, 1))
        rot = obj_init_rot_inv @ obj_rot_batch
        eul = matrix_to_euler_angles(rot)
        pos_err = torch.norm(self.pos_weight * trans.squeeze(-1), dim=-1)
        ori_err = torch.norm(self.ori_weight * eul, dim=-1)
        ori_err[ori_err < self.convergence_val[0]] = 0.0
        pos_err[pos_err < self.convergence_val[1]] = 0.0
        cost = self.weight[0] * self.orientation_gaussian(ori_err) + self.weight[1] * self.position_gaussian(pos_err)

        return cost.to(inp_device), ori_err, pos_err


