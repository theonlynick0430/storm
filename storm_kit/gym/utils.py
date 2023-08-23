from ..differentiable_robot_model.coordinate_transform import _copysign
import torch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_conjugate, quat_mul
import copy

def transform_to_torch(transform, tensor_args):
    mat = torch.eye(4, **tensor_args).unsqueeze(0)
    quat = torch.tensor([[transform.r.x, transform.r.y, transform.r.z, transform.r.w]], **tensor_args)
    rot_mat = quat2mat(quat)
    mat[0, 0,3] = transform.p.x
    mat[0, 1,3] = transform.p.y
    mat[0, 2,3] = transform.p.z
    mat[0, :3,:3] = rot_mat
    return mat

def torch_to_transform(mat):
    vec3 = torch_to_vec3(mat[0, :3, 3])
    quat = torch_to_quat(mat2quat(mat[:, :3, :3]).squeeze(0))
    return gymapi.Transform(vec3, quat)

def torch_to_quat(arr):
    return gymapi.Quat(arr[0], arr[1], arr[2], arr[3]) # xyzw format

def torch_to_vec3(arr):
    return gymapi.Vec3(arr[0], arr[1], arr[2])

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def pose_from(pos, quat, tensor_args):
    rot_mat = quat2mat(quat)
    pose = torch.eye(4, **tensor_args).unsqueeze(0).expand(pos.shape[0], -1, -1)
    pose[:, :3, :3] = rot_mat
    pose[:, :3, 3] = pos
    return pose

def quat2mat(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def mat2quat(matrix):
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4). [qx,qy,qz,qw]
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    zero = matrix.new_zeros((1,))
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    o0 = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 + m11 + m22))
    x = 0.5 * torch.sqrt(torch.max(zero, 1 + m00 - m11 - m22))
    y = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 + m11 - m22))
    z = 0.5 * torch.sqrt(torch.max(zero, 1 - m00 - m11 + m22))
    o1 = _copysign(x, matrix[..., 2, 1] - matrix[..., 1, 2])
    o2 = _copysign(y, matrix[..., 0, 2] - matrix[..., 2, 0])
    o3 = _copysign(z, matrix[..., 1, 0] - matrix[..., 0, 1])
    return torch.stack((o1, o2, o3, o0), -1)