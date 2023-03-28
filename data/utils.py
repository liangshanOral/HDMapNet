import numpy as np

import torch


def get_proj_mat(intrins, rots, trans):
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT
#和cv学的对上了，好奇妙(ps:三维坐标写成齐次式，所以维度是4)


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    #相机坐标系转成了像素坐标系
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords

'''One-hot编码是一种将分类数据表示为数值数据的方法。它涉及将具有n个可能值的分类变量转换为n个二进制变量，每个变量表示原始变量的一个可能值。
例如，如果我们有一个具有三个可能值的分类变量：“红色”，“绿色”和“蓝色”，我们可以使用三个二进制变量来表示它：[1,0,0]表示“红色”，[0,1,0]表示“绿色”，[0,0,1]表示“蓝色”。'''

def label_onehot_decoding(onehot):
    return torch.argmax(onehot, axis=0)


def label_onehot_encoding(label, num_classes=4):
    H, W = label.shape
    onehot = torch.zeros((num_classes, H, W))
    onehot.scatter_(0, label[None].long(), 1)
    return onehot


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx
