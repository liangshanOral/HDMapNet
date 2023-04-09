#import os
import numpy as np

import torch
from PIL import Image
from pyquaternion import Quaternion
from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from torch.utils.data import Dataset
from data.rasterize import preprocess_map
from .const import CAMS, NUM_CLASSES, IMG_ORIGIN_H, IMG_ORIGIN_W
from .vector_map import VectorizedLocalMap
from .lidar import get_lidar_data
from .image import normalize_img, img_transform
from .utils import label_onehot_encoding
from model.voxel import pad_or_trim_to_np

#核心就是把每张图片的信息输出，包括内外参矩阵，语义掩码等等
class HDMapNetDataset(Dataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetDataset, self).__init__()
        patch_h = data_conf['ybound'][1] - data_conf['ybound'][0]
        patch_w = data_conf['xbound'][1] - data_conf['xbound'][0]
        canvas_h = int(patch_h / data_conf['ybound'][2])
        canvas_w = int(patch_w / data_conf['xbound'][2])
        '''这两行代码计算了画布的高度和宽度。`patch_h` 和 `patch_w` 分别表示 patch 的高度和宽度，`data_conf['ybound'][2]` 和 `data_conf['xbound'][2]` 分别表示 y 轴和 x 轴的分辨率。

画布的高度和宽度是通过将 patch 的高度和宽度除以相应轴的分辨率来计算的。这样，画布的大小就可以根据 patch 的大小和分辨率自动调整。

画布用于绘制地图。在 HDMapNet 模型中，地图被表示为一个二维数组，其中每个元素表示地图上一个小块区域的语义类别。画布的大小决定了地图的分辨率。'''
        
        self.is_train = is_train
        self.data_conf = data_conf
        self.patch_size = (patch_h, patch_w)
        self.canvas_size = (canvas_h, canvas_w)
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
        self.vector_map = VectorizedLocalMap(dataroot, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.scenes = self.get_scenes(version, is_train)
        self.samples = self.get_samples()

    def __len__(self):
        return len(self.samples)

    def get_scenes(self, version, is_train):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[version][is_train]

        return create_splits_scenes()[split]

    def get_samples(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples

    def get_lidar(self, rec):
        lidar_data = get_lidar_data(self.nusc, rec, nsweeps=3, min_distance=2.2)
        #激光雷达可以快速地进行多次扫描，从而获取周围环境的三维点云数据
        lidar_data = lidar_data.transpose(1, 0)
        #(5,n)-(n,5) 5: x y z reflection rate other
        num_points = lidar_data.shape[0]
        lidar_data = pad_or_trim_to_np(lidar_data, [81920, 5]).astype('float32')
        lidar_mask = np.ones(81920).astype('float32')
        lidar_mask[num_points:] *= 0.0
        #将掩码数组中多余的元素设为 0。这一步是为了保证掩码数组中只有前 num_points 个元素为 1，其余元素都为 0。
        return lidar_data, lidar_mask

    def get_ego_pose(self, rec):
        sample_data_record = self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
        #姿态信息
        car_trans = ego_pose['translation']
        #位置信息
        pos_rotation = Quaternion(ego_pose['rotation'])
        #创建四元数表示ego旋转信息
        yaw_pitch_roll = pos_rotation.yaw_pitch_roll
        #计算欧拉角
        return torch.tensor(car_trans), torch.tensor(yaw_pitch_roll)

    def sample_augmentation(self):
        fH, fW = self.data_conf['image_size']
        resize = (fW / IMG_ORIGIN_W, fH / IMG_ORIGIN_H)
        resize_dims = (fW, fH)
        return resize, resize_dims
        #用来resizede

    # def sample_augmentation(self):
    #     self.data_conf['resize_lim'] = (0.193, 0.225)
    #     self.data_conf['bot_pct_lim'] = (0.0, 0.22)
    #     self.data_conf['rand_flip'] = True
    #     self.data_conf['rot_lim'] = (-5.4, -5.4)

    #     fH, fW = self.data_conf['image_size']
    #     if self.is_train:
    #         resize = np.random.uniform(*self.data_conf['resize_lim'])
    #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
    #         newW, newH = resize_dims
    #         crop_h = int((1 - np.random.uniform(*self.data_conf['bot_pct_lim']))*newH) - fH
    #         crop_w = int(np.random.uniform(0, max(0, newW - fW)))
    #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    #         flip = False
    #         if self.data_conf['rand_flip'] and np.random.choice([0, 1]):
    #             flip = True
    #         rotate = np.random.uniform(*self.data_conf['rot_lim'])
    #     else:
    #         resize = max(fH/IMG_ORIGIN_H, fW/IMG_ORIGIN_W)
    #         resize_dims = (int(IMG_ORIGIN_W*resize), int(IMG_ORIGIN_H*resize))
    #         newW, newH = resize_dims
    #         crop_h = int((1 - np.mean(self.data_conf['bot_pct_lim']))*newH) - fH
    #         crop_w = int(max(0, newW - fW) / 2)
    #         crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    #         flip = False
    #         rotate = 0
    #     return resize, resize_dims, crop, flip, rotate


    def get_imgs(self, rec):
        imgs = []
        trans = []
        rots = []
        intrins = []
        post_trans = []
        post_rots = []

        for cam in CAMS:
            samp = self.nusc.get('sample_data', rec['data'][cam])
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)

            resize, resize_dims = self.sample_augmentation()
            img, post_rot, post_tran = img_transform(img, resize, resize_dims)
            # resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img, post_rot, post_tran = img_transform(img, resize, resize_dims, crop, flip, rotate)

            img = normalize_img(img)
            post_trans.append(post_tran)
            post_rots.append(post_rot)
            imgs.append(img)

            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            #校准传感器
            trans.append(torch.Tensor(sens['translation']))
            rots.append(torch.Tensor(Quaternion(sens['rotation']).rotation_matrix))
            intrins.append(torch.Tensor(sens['camera_intrinsic']))
            #内参矩阵
        return torch.stack(imgs), torch.stack(trans), torch.stack(rots), torch.stack(intrins), torch.stack(post_trans), torch.stack(post_rots)

    def get_vectors(self, rec):
        location = self.nusc.get('log', self.nusc.get('scene', rec['scene_token'])['log_token'])['location']
        #场景位置
        ego_pose = self.nusc.get('ego_pose', self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        #车辆具体信息
        vectors = self.vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        return vectors

    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        vectors = self.get_vectors(rec)

        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, vectors


class HDMapNetSemanticDataset(HDMapNetDataset):
    def __init__(self, version, dataroot, data_conf, is_train):
        super(HDMapNetSemanticDataset, self).__init__(version, dataroot, data_conf, is_train)
        self.thickness = data_conf['thickness']
        self.angle_class = data_conf['angle_class']

    def get_semantic_map(self, rec):
        vectors = self.get_vectors(rec)
        instance_masks, forward_masks, backward_masks = preprocess_map(vectors, self.patch_size, self.canvas_size, NUM_CLASSES, self.thickness, self.angle_class)
        semantic_masks = instance_masks != 0
        semantic_masks = torch.cat([(~torch.any(semantic_masks, axis=0)).unsqueeze(0), semantic_masks])
        instance_masks = instance_masks.sum(0)
        forward_oh_masks = label_onehot_encoding(forward_masks, self.angle_class+1)
        backward_oh_masks = label_onehot_encoding(backward_masks, self.angle_class+1)
        direction_masks = forward_oh_masks + backward_oh_masks
        direction_masks = direction_masks / direction_masks.sum(0)
        return semantic_masks, instance_masks, forward_masks, backward_masks, direction_masks
    '''vector被转化为语义分割的掩码，实例掩码指定每个像素属于哪个实例，方向掩码由前向+后向构成
    掩码可用来训练语义分割模型
    one-hot编码是指将一个长度为N的离散型向量表示为一个长度为N的二进制向量，该二进制向量中只有一位为1，其他位都为0。
    方便处理，可解释
    
    def __getitem__(self, idx):
        rec = self.samples[idx]
        imgs, trans, rots, intrins, post_trans, post_rots = self.get_imgs(rec)
        lidar_data, lidar_mask = self.get_lidar(rec)
        car_trans, yaw_pitch_roll = self.get_ego_pose(rec)
        semantic_masks, instance_masks, _, _, direction_masks = self.get_semantic_map(rec)
        return imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_masks


def semantic_dataset(version, dataroot, data_conf, bsz, nworkers):
    train_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=True)
    val_dataset = HDMapNetSemanticDataset(version, dataroot, data_conf, is_train=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bsz, shuffle=True, num_workers=nworkers, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bsz, shuffle=False, num_workers=nworkers)
    return train_loader, val_loader


if __name__ == '__main__':
    data_conf = {
        'image_size': (900, 1600),
        'xbound': [-30.0, 30.0, 0.15],
        'ybound': [-15.0, 15.0, 0.15],
        'thickness': 5,
        'angle_class': 36,
    }

    dataset = HDMapNetSemanticDataset(version='v1.0-mini', dataroot='dataset/nuScenes', data_conf=data_conf, is_train=False)
    for idx in range(dataset.__len__()):
        imgs, trans, rots, intrins, post_trans, post_rots, lidar_data, lidar_mask, car_trans, yaw_pitch_roll, semantic_masks, instance_masks, direction_mask = dataset.__getitem__(idx)

