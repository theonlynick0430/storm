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
import torch.autograd.profiler as profiler
import copy

from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ..cost.traj_cost import TrajectoryCost
from ...mpc.rollout.arm_base import ArmBase

class ArmTrajectory(ArmBase):
    """
    This rollout function is for following a trajectory of poses for a robot

    Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        super(ArmTrajectory, self).__init__(exp_params=exp_params,
                                         tensor_args=tensor_args,
                                         world_params=world_params)
        self.ee_goal_pos_traj = None
        self.ee_goal_rot_traj = None
        self.active_idx = 0
        self.traj_length = 0

        self.goal_cost = TrajectoryCost(**exp_params['cost']['goal_pose'],
                                  tensor_args=self.tensor_args)
        

    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True, return_dist=False):

        cost = super(ArmTrajectory, self).cost_fn(state_dict, action_batch, no_coll, horizon_cost)
        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        state_batch = state_dict['state_seq']
        
        goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward(ee_pos_batch, ee_rot_batch,
                                                                    copy.deepcopy(self.ee_goal_pos_traj[self.active_idx:]), 
                                                                    copy.deepcopy(self.ee_goal_rot_traj[self.active_idx:]))
        cost += goal_cost

        if(return_dist):
            return cost, rot_err_norm, goal_dist
  
        if self.exp_params['cost']['zero_acc']['weight'] > 0:
            cost += self.zero_acc_cost.forward(state_batch[:, :, self.n_dofs*2:self.n_dofs*3], goal_dist=goal_dist.unsqueeze(-1))

        if self.exp_params['cost']['zero_vel']['weight'] > 0:
            cost += self.zero_vel_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs*2], goal_dist=goal_dist.unsqueeze(-1))
        
        return cost


    def update_params(self, retract_state=None, active_idx=None, ee_goal_pos_traj=None, ee_goal_rot_traj=None):
        """
        Update params for the cost terms and dynamics model.
        ee_goal_pos_traj: (horizon, 3)
        ee_goal_rot_traj: (horizon, 3, 3)
        """
        
        super(ArmTrajectory, self).update_params(retract_state=retract_state)

        if active_idx is not None:
            self.active_idx = min(self.traj_length-1, active_idx)
        if ee_goal_pos_traj is not None:
            self.ee_goal_pos_traj = torch.as_tensor(ee_goal_pos_traj, **self.tensor_args)
            self.traj_length = ee_goal_pos_traj.shape[0]
        if ee_goal_rot_traj is not None:
            self.ee_goal_rot_traj = torch.as_tensor(ee_goal_rot_traj, **self.tensor_args)
        
        return True
    
