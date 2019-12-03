#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
SSC DataLoader
---
Jie Li
jieli_cn@163.com
Nanjing University of Science and Technology
University of Adelaide
18/11/2018
"""
import glob
import numpy as np
import numpy.matlib
import torch.utils.data
# from scipy import misc
import imageio
from torchvision import transforms
# import datetime
""" 
1.注意 Channel first 和 Channel last 
2.注意 D H W 和 W H D 的顺序，

D H W是按照网络中3D_conv的顺序, 本文件中，除了网络的输入输出部分是D H W，其余均按照 W H D。
"""

#  data_type 统一为 float32 int32
# TODO: put these in the class
H, W = 480, 640                 # frame_height, frame_width, h x w = 480x640
voxel_SIZE = (240, 144, 240)    # 240x144x240 = 8294400
voxel_UNIT = 0.02               # 0.02m, length of each grid == 20mm
# vox_margin = 0.24
depth_T_min = 0.5
depth_T_max = 4.8


cam_K = [[518.8579, 0,        320],  # K is [fx 0 cx; 0 fy cy; 0 0 1];
         [0,        518.8579, 240],  # cx = K(1,3); cy = K(2,3);
         [0,        0,          1]]  # fx = K(1,1); fy = K(2,2);
# C_NUM = 12  # number of classes, 0 - 11, 12 classes
# 'empty','ceiling','floor','wall','window','chair','bed','sofa','table','tvs','furn','objs'
# 12, 'Accessible area'
#                0  1  2  3  4   5  6  7  8  9 10  11  12  13  14  15 16 17  18  19  20
seg_class_map = [0, 1, 2, 3, 4, 11, 5, 6, 7, 8, 8, 10, 10, 10, 11, 11, 9, 8, 11, 11, 11,
                 11, 11, 11, 11, 11, 11, 10, 10, 11, 8, 10, 11, 9, 11, 11, 11]  # 0 - 11
#                21  22  23  24  25  26  27  28  29 30  31  32 33  34  35  36


# ssc: color map
colorMap = np.array([[22, 191, 206],    # 0 empty, free space
                     [214,  38, 40],    # 1 ceiling
                     [43, 160, 4],      # 2 floor
                     [158, 216, 229],   # 3 wall
                     [114, 158, 206],   # 4 window
                     [204, 204, 91],    # 5 chair  new: 180, 220, 90
                     [255, 186, 119],   # 6 bed
                     [147, 102, 188],   # 7 sofa
                     [30, 119, 181],    # 8 table
                     [188, 188, 33],    # 9 tvs
                     [255, 127, 12],    # 10 furn
                     [196, 175, 214],   # 11 objects
                     [153, 153, 153],     # 12 label==255, ignore
                     ]).astype(np.int32)

class NYUv2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, encoding='TSDF', downsample=1, data_augment=True):
        r"""
        Arguments:
            root (str): path of folder containing sample data.
            mode (str): 'TRAIN', 'VAL', 'TEST', 'TEST', 'VIS'
            encoding (str): 'BINARY', 0 for empty, 1 for occupancy
                            'STSDF', 1 surface, 0 front and empty, -1 behind
                            'SDF', -1 <= TSDF <= 1, 1 for empty
                            'NUMBER', NUMBER of points in the bin, 0 for empty
            downsample (int): downsample the targets

        """
        mode_list = ['TRAIN', 'TRAIN_GN', 'TEST', 'PREDICT', 'TEST_TSDF', 'PREDICT_TSDF']
        encoding_list = ['RGB', 'XYZRGB', 'BINARY', 'XYZ', 'STSDF', 'TSDF']
        if root is None:
            raise Exception("Oops! 'root' is None, please set the right file path.")

        self.filepaths = list()
        if isinstance(root, list):  # 将多个root
            for root_i in root:
                fp = glob.glob(root_i + '/*.bin')
                fp.sort()
                self.filepaths.extend(fp)
        elif isinstance(root, str):
            # print('root is string ', root)
            self.filepaths = glob.glob(root + '/*.bin')  # List all files in data folder
            self.filepaths.sort()

        # np.savetxt('NYUCADtrain_filenames.txt', self.filepaths, fmt="%s")
        self.mode = mode
        self.encoding = encoding
        self.data_augment = False
        print('self.data_augment', self.data_augment)

        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] \
        # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        self.transforms_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if mode not in mode_list:
            raise Exception("Oops!  '{}' is not a  valid mode. Try {}.".format(mode, mode_list))
        if encoding not in encoding_list:
            raise Exception("Oops!  '{}' is not a valid Encoding. Try {}.".format(encoding, encoding_list))
        if len(self.filepaths) == 0:
            raise Exception("Oops!  That was no valid data in '{}'.".format(root))
        if not isinstance(downsample, int):
            raise Exception("An int is expected, but got {}".format(type(downsample)))

        self.downsample = downsample  # int, downsample = 4, in labeled data, get 1 voxel from each 4
        self.vox_size = (voxel_SIZE[0] / downsample, voxel_SIZE[1] / downsample, voxel_SIZE[2] / downsample)
        print('SuncgDataset:mode:{}, encoding:{}, {} files, Resolution:{}'.format(mode, encoding, len(self.filepaths), self.vox_size))

    def __getitem__(self, index):
        r"""
        Shape:
            voxels: (1, 240, 144, 240), pytorch needs channel-first
            target: (1, 60, 36, 60)
        """
        _name = self.filepaths[index][:-4]
        depth = self._read_depth(_name + '.png', 480, 640)  # (h, w)
        vox_origin, cam_pose, rle = self._read_rle(_name + '.bin')

        target_hr = self._rle2voxel(rle, _name + '.bin') if self.mode is not 'PREDICT' else None
        # target_lr = self._downsample_label(target_hr, 4) if self.mode is not 'PREDICT' else None

        # TODO rgb vs depth
        # binary_vox, xyz_vox = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
        # voxels = rgb_vox if self.encoding == 'RGB' else xyzrgb_vox
        # TODO xyzrgb/240 144 240 255 255 255

        if self.encoding == 'BINARY':
            binary_vox, _, _ = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
            voxels = binary_vox.reshape(binary_vox.shape + (1,))  # (W, H, D, 1)
        if self.encoding == 'XYZ':
            binary_vox, xyz_vox, _ = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
            # voxels = xyz_vox              # (W, H, D, ３), 0-240, 0-144, 0-240
            voxels = xyz_vox / voxel_SIZE   # (W, H, D, ３), 0-1.  # TODO type np.float32
        elif self.encoding == 'STSDF':
            binary_vox, _, _ = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
            stsdf_hr = self._get_stsdf(depth, binary_vox, vox_origin, cam_pose)  # (W, H, D)
            voxels = stsdf_hr.reshape(stsdf_hr.shape + (1,))   # (W, H, D, 1)
        elif self.encoding == 'TSDF':
            binary_vox, _, position = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
            npz_file = np.load(_name + '.npz')
            # target_lr = npz_file['label'].astype(np.int32)  # (W, H, D), SUNCG
            tsdf_hr = npz_file['tsdf']  # SUNCG (W, H, D)
            voxels = tsdf_hr.reshape(tsdf_hr.shape + (1,))  # (W, H, D, 1)
        elif self.encoding == 'RGB':
            rgb = self._read_rgb(_name + '.jpg', 480, 640)  # (h=image_height, w=image_width, 3)
            rgb = rgb / 255.0  # if normalise else rgb  # normalise rgb from [0, 255] to [0, 1]
            binary_vox, rgb_vox, xyz_vox = self._rgbd2voxel(rgb, depth, cam_pose, vox_origin, voxel_UNIT)
            voxels = rgb_vox
        elif self.encoding == 'XYZRGB':
            rgb = self._read_rgb(_name + '.jpg', 480, 640)  # (h=image_height, w=image_width, 3)
            rgb = rgb / 255.0  # if normalise else rgb  # normalise rgb from [0, 255] to [0, 1]
            binary_vox, rgb_vox, xyz_vox = self._rgbd2voxel(rgb, depth, cam_pose, vox_origin, voxel_UNIT)
            xyz_vox = xyz_vox / voxel_SIZE  # xyz_vox.shape[:3]
            voxels = np.concatenate((xyz_vox, rgb_vox), 3)  # (W, H, D, 6)

        if self.mode == 'TRAIN':  # Have labeled data
            # ---- data augment
            if self.data_augment:
                voxels, target_hr = self._data_augment(voxels, target_hr, self.encoding)

            target_lr = self._downsample_label(target_hr, 4)

            # Save GT
            # ply_filename = _name + '-HR_GT.ply'
            # self.labeled_voxel2ply(target_hr, ply_filename)

            # Save depth_voxel
            # ply_filename = _name + '-LR_depth_voxel_method2-2.ply'
            # self._depth_voxel2ply(binary_vox, ply_filename, encoding='BINARY')

            # if self.downsample == 1:  # HR
            #     one_hot_target_vox = one_hot_embedding(target_hr, 12)  # W H D C. for GAN
            #     return voxels.T, target_hr.T, target_lr.T, one_hot_target_vox.T, _name + '.png'  # C D H W, for Conv3d
            #
            # if self.downsample == 4:  # LR
            #     one_hot_target_vox = one_hot_embedding(target_lr, 12)  # W H D C. for GAN
            #     return voxels.T, target_lr.T, one_hot_target_vox.T, _name + '.png'  # C D H W, for Conv3d

            depth = depth.reshape((1,) + depth.shape)

            # rgb = self._read_rgb(_name + '.jpg', 480, 640)  # (h=image_height, w=image_width, 3)
            # rgb_tesnor = self.transforms_rgb(rgb)  # (C x H x W) = (3, 480, 640)

            if self.downsample == 1:  # HR
                # one_hot_target_vox = one_hot_embedding(target_hr, 12)  # W H D C. for GAN

                # gn = self._get_gradient_norm_semantic(target_hr, d=2)  # (W H D)
                # gn = gn + 0.2
                # 注意position[H, W, 3] 对应X，Y，Z，而voxels的坐标顺序为Z,Y,X

                return depth, voxels.T, target_hr.T, target_lr.T, position, _name + '.png'  # C D H W, for Conv3d

            if self.downsample == 4:  # LR
                # ply_filename = _name + '-HR_RGBvox-all.ply'
                # self.rgb_voxel2ply(rgb_vox, ply_filename)
                # one_hot_target_vox = one_hot_embedding(target_lr, 12)  # W H D C. for GAN

                # gn = self._get_gradient_norm_semantic(target_lr, d=2)  # (W H D)
                # print('np.amin(gn), np.amax(gn)', np.amin(gn), np.amax(gn))
                # self._gn_voxel2ply(gn, _name + 'gn.ply')
                # gn = gn + 1  # TODO 0.2 比较的参数
                # gn[target_lr == 255] = 0
                # return voxels.T, target_lr.T, gn.T, _name + '.png'  # C D H W, for Conv3d
                # 注意position[H, W, 3] 对应X，Y，Z，而voxels的坐标顺序为Z,Y,X
                return depth, voxels.T, target_lr.T, position, _name + '.png'  # C D H W, for Conv3d
                # return depth, rgb_tesnor, voxels.T, target_lr.T, position, _name + '.png'  # C D H W, for Conv3d
                # return rgb_tesnor, voxels.T, target_lr.T, position, _name + '.png'  # C D H W, for Conv3d

        if self.mode == 'TRAIN_GN':  # Have labeled data
            # ---- data augment
            if self.data_augment:
                voxels, target_hr = self._data_augment(voxels, target_hr, self.encoding)

            target_lr = self._downsample_label(target_hr, 4)

            if self.downsample == 1:  # HR
                gn_hr = self._get_gradient_norm_semantic(target_hr, d=2, ord=np.inf)  # (W H D)  TODO ord=1, 2, inf
                gn_hr = gn_hr + 1  #
                gn_lr = self._get_gradient_norm_semantic(target_lr, d=1, ord=np.inf)  # (W H D)
                gn_lr = gn_lr + 1  # 1 为原始权重，各类别、各体素权重均为1
                return voxels.T, target_hr.T, target_lr.T, gn_hr.T, gn_lr.T, _name + '.png'  # C D H W, for Conv3d

            if self.downsample == 4:  # LR
                # ply_filename = _name + '-HR_RGBvox-all.ply'
                # self.rgb_voxel2ply(rgb_vox, ply_filename)
                # one_hot_target_vox = one_hot_embedding(target_lr, 12)  # W H D C. for GAN
                gn_lr = self._get_gradient_norm_semantic(target_lr, d=2, ord=2)  # (W H D)
                gn_lr = gn_lr + 1  # 1 为原始权重，各类别、各体素权重均为1
                return voxels.T, target_lr.T, gn_lr.T, _name + '.png'  # C D H W, for Conv3d

        if self.mode == 'TEST':  # Have labeled data, nonempty=None
            target = self._downsample_label(target_hr, self.downsample)
            return voxels.T, target.T, _name + '.png'  # C D H W, for Conv3d

        if self.mode == 'TEST_TSDF':  # Have labeled data
            target = self._downsample_label(target_hr, self.downsample)
            # ---- STSDF
            # stsdf_hr = self._get_stsdf(depth, binary_vox, vox_origin, cam_pose)
            # stsdf = self._downsample_stsdf(stsdf_hr, self.downsample)
            # nonempty = self.get_nonempty(stsdf, 'STSDF')          # 更合理， 差别在于天花板顶上与墙后的处理
            # # nonempty = self.get_nonempty2(stsdf, target, 'STSDF')  # 这个更符合SUNCG的做法
            # ---- TSDF
            if self.encoding != 'TSDF':
                npz_file = np.load(_name + '.npz')
                tsdf_hr = npz_file['tsdf']  # SUNCG (W, H, D)
            tsdf = self._downsample_tsdf(tsdf_hr, self.downsample)
            # nonempty = self.get_nonempty(tsdf, 'TSDF')
            nonempty = self.get_nonempty2(tsdf, target, 'TSDF')  # 这个更符合SUNCG的做法
            # ---- save ply
            # ply_filename = _name + '-HR_empty-SUNCG-STSDF.ply'
            # self._depth_voxel2ply(nonempty, ply_filename, encoding='EMPTY')
            # ply_filename = _name + '-HR_binary-SUNCG.ply'
            # binary_vox, xyz_vox = self._depth2voxel(depth, cam_pose, vox_origin, unit=voxel_UNIT)
            # self._depth_voxel2ply(binary_vox, ply_filename, encoding='BINARY')
            # ply_filename = _name + '-HR_stsdf-SUNCG-surface.ply'
            # SuncgDataset._depth_voxel2ply(stsdf_hr, ply_filename, encoding='STSDF')

            depth = depth.reshape((1,) + depth.shape)
            # rgb = self._read_rgb(_name + '.jpg', 480, 640)  # (h=image_height, w=image_width, 3)
            # rgb_tesnor = self.transforms_rgb(rgb)  # (C x H x W) = (3, 480, 640)
            # return depth, rgb_tesnor, voxels.T, target.T, nonempty.T, position, _name + '.png'  # C D H W, for Conv3d
            # return rgb_tesnor, voxels.T, target.T, nonempty.T, position, _name + '.png'
            return depth, voxels.T, target.T, nonempty.T, position, _name + '.png'

        if self.mode == 'PREDICT':  # Do not have labeled data, only show predict results
            return voxels.T, _name + '.png'  # C D H W, for Conv3d

        if self.mode == 'PREDICT_TSDF':  # Have labeled data
            stsdf_hr = self._get_stsdf(depth, binary_vox, vox_origin, cam_pose)
            stsdf = self._downsample_stsdf(stsdf_hr, self.downsample)
            nonempty = self.get_nonempty(stsdf, 'STSDF')
            # nonempty = self.get_nonempty2(stsdf, target, 'STSDF')
            return voxels.T, nonempty.T, _name + '.png'  # C D H W, for Conv3d

    def __len__(self):
        return len(self.filepaths)

    @classmethod
    def _data_augment(cls, voxels, target, encoding):
        # encoding == 'RGB', 'XYZRGB', 'BINARY', 'XYZ', 'STSDF': _empty = 0
        # encoding == 'TSDF': _empty = np.float32(0.001)
        _empty = np.float32(0.001) if encoding == 'TSDF' else 0

        if np.random.rand() < 0.8:  # 沿Z轴(对应D),往后(Z变大的方向)平移, 最多平移Z轴的20%，平移后，空出来的部分置为空
            sz = int(np.random.rand() * voxels.shape[2] * 0.2) + 1  # at least, move 1 grid
            voxels = np.roll(voxels, shift=sz, axis=2)  # Roll array elements along axis Z.
            target = np.roll(target, shift=sz, axis=2)
            voxels[:, :, :sz, :] = _empty
            target[:, :, :sz] = 255  # 0, empty --> 255, ignore

        if np.random.rand() < 0.8:  # 沿x轴(对应W),往x变大的方向平移, 最多平移X轴的10%，平移后，空出来的部分置为空
            move_opposite = True if np.random.rand() < 0.5 else False  # move direction
            sx_1 = int(np.random.rand() * voxels.shape[0] * 0.1) + 1  # at least, move 1 grid
            sx_1 = voxels.shape[0] - sx_1 if move_opposite else sx_1
            voxels = np.roll(voxels, shift=sx_1, axis=0)  # Roll array elements along axis X.
            target = np.roll(target, shift=sx_1, axis=0)
            if move_opposite:  # 往x变小的方向平移
                voxels[sx_1:, :, :, :] = _empty
                target[sx_1:, :, :] = 255
            else:  # 往x变大的方向平移
                voxels[:sx_1, :, :, :] = _empty
                target[:sx_1, :, :] = 255

        if np.random.rand() < 0.5:  # 左右反转
            target = np.flip(target, 0).copy()  # Flip an array vertically (axis=0).
            voxels = np.flip(voxels, 0).copy()
        return voxels, target

    @staticmethod
    def _get_gradient_norm(data, d=1, ord=3):  # TODO d=1, ord=1 作为实验比较的参数
        """ 依据GT计算每个点的梯度，用梯度的大小作为loss的权重"""  #
        # data [W, H, D], Ground truth, array_like. An N-dimensional array containing samples of a scalar function.
        # d: single scalar to specify a sample distance for all dimensions.
        f = np.zeros(data.shape, dtype=np.float32)
        f[data > 0] = 1
        g = np.gradient(f, d)               # list of ndarray, (3, [W, H, D])
        g = np.asarray(g)                   # [3, W, H, D]
        gn = np.linalg.norm(g, ord=ord, axis=0)      # [W, H, D]
        return gn

    @staticmethod
    def _get_gradient_norm_semantic(data, d=1, ord=2):  # TODO d=1, ord=1 作为实验比较的参数
        """ 依据GT计算每个点的梯度，用梯度的大小作为loss的权重,
        d:  single scalar to specify a sample distance for all dimensions.
        ord: Order of the norm
        """  #
        # data [W, H, D]
        c = 12
        f = np.zeros(data.shape, dtype=np.float32)
        gn = np.zeros(data.shape, dtype=np.float32)
        # data_tmp = data
        data_tmp = np.copy(data)
        data_tmp[data == 255] = 0  # ingnore_index=255, treat these voxels as free space
        for idx in range(c):
            f.fill(0)
            f[data_tmp == idx] = 1
            g = np.gradient(f, d)               # list of ndarray, (3, [W, H, D])
            g = np.asarray(g)                   # [3, W, H, D]
            gn += np.linalg.norm(g, ord=ord, axis=0)      # [W, H, D]
        # print(type(gn), type(gn[0,0,0]))
        return gn

    @staticmethod
    def _read_depth(depth_filename, img_h=480, img_w=640):
        r"""Read a depth image with size H x W
        and save the depth values (in millimeters) into a 2d numpy array.
        The depth image file is assumed to be in 16-bit PNG format, depth in millimeters.
        """
        # depth = misc.imread(depth_filename) / 8000.0  # numpy.float64
        # depth = misc.imresize(depth, (img_h, img_w))  # numpy.uint8
        depth = imageio.imread(depth_filename) / 8000.0  # numpy.float64
        depth = np.asarray(depth)
        assert depth.shape == (img_h, img_w), 'incorrect default size'
        return depth

    @staticmethod
    def _read_rgb(rgb_filename, img_h=480, img_w=640):  # 0.01s
        r"""Read a RGB image with size H x W
        """
        # rgb = misc.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        # rgb = misc.imresize(rgb, (img_h, img_w))  # (H, W, 3)
        rgb = imageio.imread(rgb_filename)  # <type 'numpy.ndarray'>, numpy.uint8, (480, 640, 3)
        rgb = np.asarray(rgb)
        # rgb = np.rollaxis(rgb, 2, 0)  # (H, W, 3)-->(3, H, W)
        return rgb

    @staticmethod
    def _read_rle(rle_filename):  # 0.0005s
        r"""Read RLE compression data
        Return:
            vox_origin,
            cam_pose,
            vox_rle, voxel label data from file
        Shape:
            vox_rle, (240, 144, 240), if downsample==4, then (60, 36, 60)
        """
        fid = open(rle_filename, 'rb')
        vox_origin = np.fromfile(fid, np.float32, 3).T  # Read voxel origin in world coordinates
        cam_pose = np.fromfile(fid, np.float32, 16).reshape((4, 4))  # Read camera pose
        vox_rle = np.fromfile(fid, np.uint32).reshape((-1, 1)).T  # Read voxel label data from file
        vox_rle = np.squeeze(vox_rle)  # 2d array: (1 x N), to 1d array: (N , )
        fid.close()
        return vox_origin, cam_pose, vox_rle

    @staticmethod
    def _get_xyz(size=voxel_SIZE):
        """x 水平 y高低  z深度"""
        _x = np.zeros(size, dtype=np.int32)
        _y = np.zeros(size, dtype=np.int32)
        _z = np.zeros(size, dtype=np.int32)

        for i_h in range(size[0]):  # x, y, z
            _x[i_h, :, :] = i_h                 # x, left-right flip
        for i_w in range(size[1]):
            _y[:, i_w, :] = i_w                 # y, up-down flip
        for i_d in range(size[2]):
            _z[:, :, i_d] = i_d                 # z, front-back flip
        return _x, _y, _z

    @classmethod
    def _get_stsdf(cls, depth, voxel_binary, vox_origin, cam_pose):
        r"""simplified TSDF, encoding == 'STSDF'
        Shape:
            depth: (H, W)
            voxel_binary: (240, 144, 240)
            voxel_stsdf: (240, 144, 240)
        """
        c = cam_pose
        p_base = np.zeros(voxel_SIZE + (3,), dtype=np.float32)  # points
        p_cam2 = np.zeros(voxel_SIZE + (3,), dtype=np.float32)  # points in camera view
        pixel_xy = np.zeros(voxel_SIZE + (2,), dtype=np.int32)
        _x, _y, _z = cls._get_xyz()
        p_base[:, :, :, 0] = _x * voxel_UNIT + vox_origin[0]
        p_base[:, :, :, 1] = _z * voxel_UNIT + vox_origin[1]
        p_base[:, :, :, 2] = _y * voxel_UNIT + vox_origin[2]

        p_base[:, :, :, 0] = p_base[:, :, :, 0] - c[0][3]
        p_base[:, :, :, 1] = p_base[:, :, :, 1] - c[1][3]
        p_base[:, :, :, 2] = p_base[:, :, :, 2] - c[2][3]

        p_cam2[:, :, :, 0] = c[0][0] * p_base[:, :, :, 0] + c[1][0] * p_base[:, :, :, 1] + c[2][0] * p_base[:, :, :, 2]
        p_cam2[:, :, :, 1] = c[0][1] * p_base[:, :, :, 0] + c[1][1] * p_base[:, :, :, 1] + c[2][1] * p_base[:, :, :, 2]
        p_cam2[:, :, :, 2] = c[0][2] * p_base[:, :, :, 0] + c[1][2] * p_base[:, :, :, 1] + c[2][2] * p_base[:, :, :, 2]

        # NYUCADtrain NYU0601_0000.png, p_cam2[19 99  7  2]==0
        # NYUCADtest NYU0761_0000.png
        if np.count_nonzero(p_cam2[:, :, :, 2] == 0):
            tt_idx = np.nonzero(p_cam2[:, :, :, 2] == 0)
            tt_idx = np.stack(tt_idx, axis=1)
            b = np.ones((tt_idx.shape[0], 1), dtype=np.int32) * 2
            tt_idx = np.concatenate((tt_idx, b), axis=1)
            p_cam2[tt_idx] += 0.0001
            # print(f, '/0 bug')
        pixel_xy[:, :, :, 0] = np.round(cam_K[0][0] * (p_cam2[:, :, :, 0] / p_cam2[:, :, :, 2]) + cam_K[0][2])
        pixel_xy[:, :, :, 1] = np.round(cam_K[1][1] * (p_cam2[:, :, :, 1] / p_cam2[:, :, :, 2]) + cam_K[1][2])

        # initial to empty
        # 1 ----- 0 empty
        STSDF_EMPTY, STSDF_SURFACE, STSDF_OCCLUD = 0, 1, -1
        voxel_stsdf = np.zeros(voxel_SIZE, dtype=np.float32)
        # 2 ----- 1 empty
        # STSDF_EMPTY, STSDF_SURFACE, STSDF_OCCLUD = 1, 0, -1
        # voxel_stsdf = np.ones(voxel_SIZE, dtype=np.float32)

        idx_h = np.rint(pixel_xy[:, :, :, 1]).astype(np.int32)
        idx_w = np.rint(pixel_xy[:, :, :, 0]).astype(np.int32)
        idx_h[idx_h >= 480] = 0             # 超出场景范围
        idx_w[idx_w >= 640] = 0
        idx_h[idx_h < 0] = 0
        idx_w[idx_w < 0] = 0
        pt_depth = depth[idx_h, idx_w]

        voxel_stsdf[pt_depth[:] <= p_cam2[:, :, :, 2]] = STSDF_OCCLUD     # 被遮挡
        # voxel_stsdf[abs(pt_depth - p_cam2[:, :, :, 2]) < 0.0001] = 1  # 超出搜索范围
        voxel_stsdf[np.round(pt_depth) == 0] = STSDF_EMPTY
        # if pt_depth < depth_T_min or pt_depth > depth_T_max:          # 超出场景范围
        #     voxel_stsdf[x, y, z] = 1
        idx_h = pixel_xy[:, :, :, 1].astype(np.int32)
        idx_w = pixel_xy[:, :, :, 0].astype(np.int32)
        voxel_stsdf[idx_h >= 480] = STSDF_EMPTY                 # 超出场景范围
        voxel_stsdf[idx_w >= 640] = STSDF_EMPTY
        voxel_stsdf[idx_h < 0] = STSDF_EMPTY
        voxel_stsdf[idx_w < 0] = STSDF_EMPTY
        voxel_stsdf[p_cam2[:, :, :, 2] <= 0] = STSDF_EMPTY      # 视野之外
        voxel_stsdf[voxel_binary > 0] = STSDF_SURFACE           # surface

        # 1 ----- 0 empty
        # voxel_stsdf = 1 - voxel_stsdf     # surface, empty, occulted: 1, 0, -1 ---> 0 , 1 , -2
        # 2 ----- 1 empty
        # voxel_stsdf = 1 - voxel_stsdf     # surface, empty, occulted: 0, 1, -1 ---> 1 , 0 , -2
        del p_base, p_cam2, pixel_xy
        return voxel_stsdf

    @classmethod
    def _rle2voxel(cls, rle, rle_filename=''):
        r"""Read voxel label data from file (RLE compression), and convert it to fully occupancy labeled voxels.
        In the data loader of pytorch, only single thread is allowed.
        For multi-threads version and more details, see 'readRLE.py'.
        output: seg_label: 3D numpy array, size 240 x 144 x 240
        """
        # ---- Read RLE
        # vox_origin, cam_pose, rle = cls._read_rle(rle_filename)
        # ---- Uncompress RLE, 0.9s
        seg_label = np.zeros(voxel_SIZE[0] * voxel_SIZE[1] * voxel_SIZE[2], dtype=np.uint8)  # segmentation label
        vox_idx = 0
        for idx in range(int(rle.shape[0] / 2.0)):
            check_val = rle[idx * 2]
            check_iter = rle[idx * 2 + 1]
            if check_val >= 37 and check_val != 255:  # 37 classes to 12 classes
                print('RLE {} check_val: {}'.format(rle_filename, check_val))
            # seg_label_val = 1 if check_val < 37 else 0  # 37 classes to 2 classes: empty or occupancy
            # seg_label_val = 255 if check_val == 255 else seg_class_map[check_val]
            seg_label_val = seg_class_map[check_val] if check_val != 255 else 255  # 37 classes to 12 classes
            seg_label[vox_idx: vox_idx+check_iter] = np.matlib.repmat(seg_label_val, 1, check_iter)
            vox_idx = vox_idx + check_iter
        seg_label = seg_label.reshape(voxel_SIZE)  # 3D array, size 240 x 144 x 240
        return seg_label

    @classmethod  # method 1
    def _depth2voxel_old(cls, depth, cam_pose, vox_origin, unit=0.02):
        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_K[0][2]) * depth / cam_K[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_K[1][2]) * depth / cam_K[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        # pt_world2 = pt_world
        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        # pt_world2[:, :, 0] = pt_world[:, :, 1]  # x 原始paper方法
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        # pt_world2[:, :, 2] = pt_world[:, :, 0]  # z

        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(np.int32).reshape((-1, 3))  # (h*w, 3)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_SIZE, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros(voxel_SIZE, dtype=np.float32)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_SIZE + (3,), dtype=np.float32)  # (W, H, D, 3)
        for i_idx in range(len(point_grid)):
            i_x, i_y, i_z = point_grid[i_idx, :]
            # i_x, i_y, i_z = int(i_x), int(i_y), int(i_z)
            if i_x < voxel_SIZE[0] and i_y < voxel_SIZE[1] and i_z < voxel_SIZE[2] \
                    and i_x >= 0 and i_y >= 0 and i_z >= 0:
                voxel_binary[i_x][i_y][i_z] = 1  # the bin has at least one point (bin is not empty)
                voxel_xyz[i_x, i_y, i_z, :] = point_grid[i_idx, :]
        # output --- 3D Tensor, 240 x 144 x 240
        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid     # Release Memory
        return voxel_binary, voxel_xyz   # (W, H, D), (W, H, D, 3)

    @classmethod  # method 2, new
    def _depth2voxel(cls, depth, cam_pose, vox_origin, unit=0.02):
        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_K[0][2]) * depth / cam_K[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_K[1][2]) * depth / cam_K[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        # pt_world2 = pt_world
        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        # pt_world2[:, :, 0] = pt_world[:, :, 1]  # x 原始paper方法
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        # pt_world2[:, :, 2] = pt_world[:, :, 0]  # z

        """
        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(np.int32).reshape((-1, 3))  # (h*w, 3)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_SIZE, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros(voxel_SIZE, dtype=np.float32)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_SIZE + (3,), dtype=np.float32)  # (W, H, D, 3)
        position = np.zeros((H, W, 3), dtype=np.int32)
        for i_idx in range(len(point_grid)):
            i_x, i_y, i_z = point_grid[i_idx, :]
            # i_x, i_y, i_z = int(i_x), int(i_y), int(i_z)
            if i_x < voxel_SIZE[0] and i_y < voxel_SIZE[1] and i_z < voxel_SIZE[2] \
                    and i_x >= 0 and i_y >= 0 and i_z >= 0:
                voxel_binary[i_x][i_y][i_z] = 1  # the bin has at least one point (bin is not empty)
                voxel_xyz[i_x, i_y, i_z, :] = point_grid[i_idx, :]
                h = i_idx / W
                w = i_idx - h * W
                position[h, w, :] = point_grid[i_idx, :]  # 记录图片上的每个像素对应的voxel位置
        """
        # ---- World coordinate to grid/voxel coordinate
        point_grid = pt_world2 / unit  # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(np.int32)  # .reshape((-1, 3))  # (H*W, 3) (H, W, 3)
        # print(point_grid.shape)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        # voxel_binary = np.zeros(voxel_SIZE, dtype=np.uint8)     # (W, H, D)
        voxel_binary = np.zeros(voxel_SIZE, dtype=np.float32)  # (W, H, D)
        voxel_xyz = np.zeros(voxel_SIZE + (3,), dtype=np.float32)  # (W, H, D, 3)
        # position = np.zeros((H, W, 3), dtype=np.int32)
        position = np.zeros((H, W), dtype=np.int32)

        # [rows, cols, c] = point_grid.shape  # (H, W, 3)
        # rows = H
        # cols = W
        for h in range(H):
            for w in range(W):
                i_x, i_y, i_z = point_grid[h, w, :]
                if 0 <= i_x < voxel_SIZE[0] and 0 <= i_y < voxel_SIZE[1] and 0 <= i_z < voxel_SIZE[2]:
                    voxel_binary[i_x][i_y][i_z] = 1  # the bin has at least one point (bin is not empty)
                    voxel_xyz[i_x, i_y, i_z, :] = point_grid[h, w, :]
                    # position[h, w, :] = point_grid[h, w, :]  # 记录图片上的每个像素对应的voxel位置
                    # idx = np.ravel_multi_index(point_grid[h, w, :], voxel_SIZE)
                    position[h, w] = np.ravel_multi_index(point_grid[h, w, :], voxel_SIZE)  # 记录图片上的每个像素对应的voxel位置
                # print(num[h, w])

        # output --- 3D Tensor, 240 x 144 x 240
        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid     # Release Memory
        # position[h,w]
        return voxel_binary, voxel_xyz, position   # (W, H, D), (W, H, D, 3)

    @classmethod
    def _rgbd2voxel(cls, rgb, depth, cam_pose, vox_origin, unit=0.02):
        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(W), range(H))
        pt_cam = np.zeros((H, W, 3), dtype=np.float32)
        pt_cam[:, :, 0] = (gx - cam_K[0][2]) * depth / cam_K[0][0]  # x
        pt_cam[:, :, 1] = (gy - cam_K[1][2]) * depth / cam_K[1][1]  # y
        pt_cam[:, :, 2] = depth  # z, in meter
        # ---- Get point in world coordinate
        p = cam_pose
        pt_world = np.zeros((H, W, 3), dtype=np.float32)
        pt_world[:, :, 0] = p[0][0] * pt_cam[:, :, 0] + p[0][1] * pt_cam[:, :, 1] + p[0][2] * pt_cam[:, :, 2] + p[0][3]
        pt_world[:, :, 1] = p[1][0] * pt_cam[:, :, 0] + p[1][1] * pt_cam[:, :, 1] + p[1][2] * pt_cam[:, :, 2] + p[1][3]
        pt_world[:, :, 2] = p[2][0] * pt_cam[:, :, 0] + p[2][1] * pt_cam[:, :, 1] + p[2][2] * pt_cam[:, :, 2] + p[2][3]
        pt_world[:, :, 0] = pt_world[:, :, 0] - vox_origin[0]
        pt_world[:, :, 1] = pt_world[:, :, 1] - vox_origin[1]
        pt_world[:, :, 2] = pt_world[:, :, 2] - vox_origin[2]
        # ---- Aline the coordinates with labeled data (RLE .bin file)
        pt_world2 = np.zeros(pt_world.shape, dtype=np.float32)  # (h, w, 3)
        # pt_world2 = pt_world
        pt_world2[:, :, 0] = pt_world[:, :, 0]  # x 水平
        pt_world2[:, :, 1] = pt_world[:, :, 2]  # y 高低
        pt_world2[:, :, 2] = pt_world[:, :, 1]  # z 深度

        # pt_world2[:, :, 0] = pt_world[:, :, 1]  # x 原始paper方法
        # pt_world2[:, :, 1] = pt_world[:, :, 2]  # y
        # pt_world2[:, :, 2] = pt_world[:, :, 0]  # z

        # ---- World coordinate to grid/voxel coordinate
        # point_grid = pt_world2 / voxel_UNIT           # Get point in grid coordinate, each grid is a voxel
        point_grid = pt_world2 / unit                   # Get point in grid coordinate, each grid is a voxel
        point_grid = np.rint(point_grid).astype(int).reshape((-1, 3))   # (h*w, 3)
        # point_grid = point_grid.reshape((-1, 3))                      # (h*w, 3)
        rgb = rgb.reshape((-1, 3))                                      # (h*w, 3)

        # ---- crop depth to grid/voxel
        # binary encoding '01': 0 for empty, 1 for occupancy
        voxel_binary = np.zeros(voxel_SIZE, dtype=np.float32)           # (W, H, D) dtype=np.uint8
        voxel_xyz = np.zeros(voxel_SIZE + (3,), dtype=np.float32)       # (W, H, D, 3)
        voxel_rgb = np.zeros(voxel_SIZE + (3,), dtype=np.float32)       # (W, H, D, 3)
        # voxel_xyzrgb = np.zeros(voxel_SIZE + (6,), dtype=np.float32)    # (W, H, D, 6)
        for i_idx in range(len(point_grid)):
            i_x, i_y, i_z = point_grid[i_idx, :]
            # i_x, i_y, i_z = int(i_x), int(i_y), int(i_z)
            if i_x < voxel_SIZE[0] and i_y < voxel_SIZE[1] and i_z < voxel_SIZE[2] \
                    and i_x >= 0 and i_y >= 0 and i_z >= 0:
                # if 0 <= i_x < voxel_SIZE[0] and 0 <= i_y < voxel_SIZE[1] and 0 <= i_z < voxel_SIZE[2]:
                voxel_binary[i_x][i_y][i_z] = 1  # the bin has at least one point (bin is not empty)
                voxel_xyz[i_x, i_y, i_z, :] = point_grid[i_idx, :]
                voxel_rgb[i_x, i_y, i_z, :] = rgb[i_idx, :]
                # voxel_xyzrgb[i_x, i_y, i_z, :3] = point_grid[i_idx, :]
                # voxel_xyzrgb[i_x, i_y, i_z, 3:] = rgb[i_idx, :]
        # output --- 3D Tensor, 240 x 144 x 240
        del depth, gx, gy, pt_cam, pt_world, pt_world2, point_grid     # Release Memory
        # voxel_binary = voxel_binary.reshape((1,) + voxel_binary.shape)   # channel first, (1, W, H, D)
        # voxel_binary = voxel_binary.reshape(voxel_binary.shape + (1,))   # channel last , (W, H, D, 1)
        return voxel_binary, voxel_rgb, voxel_xyz  # voxel_xyzrgb    # (W, H, D), (W, H, D, 3), (W, H, D, 6)

    @classmethod
    def _depth2ply(cls, depth, ply_filename):
        # depth = cls._read_depth(depth_filename)
        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(W), range(H))
        point_cam = np.zeros((H, W, 3), dtype=float)
        point_cam[:, :, 0] = (gx - cam_K[0][2]) * depth / cam_K[0][0]  # x
        point_cam[:, :, 1] = (gy - cam_K[1][2]) * depth / cam_K[1][1]  # y
        point_cam[:, :, 2] = depth  # z, in meter
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'end_header' % depth.size
        # ply_filename = depth_filename[:-4] + '_depth2ply.ply'
        ply_data = point_cam.reshape((-1, 3))
        np.savetxt(ply_filename, ply_data, fmt="%f %f %f", header=ply_head, comments='')
        print('Saved-->{}'.format(ply_filename))

    @classmethod
    def _depth_voxel2ply(cls, voxel_val, ply_filename, encoding='BINARY'):
        """ply: x y z of voxels from depth. only save the voxels containing points"""
        STSDF_EMPTY = np.float32(0)  # 1 is surface, -2 is occluded, 0 is free
        TSDF_EMPTY = np.float32(0.001)  # -0.001 is surface, -0.001 is occluded, 0.001 is free
        # ---- get size
        size = voxel_val.shape
        # print('size', size)
        _x, _y, _z = cls._get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        ply_data_grid = zip(_x, _y, _z, voxel_val.flatten())
        ply_data = []
        if encoding == 'BINARY':  # TODO RGB XYZ XYZRGB ...
            for i_idx in range(len(ply_data_grid)):
                if ply_data_grid[i_idx][3] > 0:  # 0 is empty
                    ply_data.append(ply_data_grid[i_idx])
        if encoding == 'STSDF':
            for i_idx in range(len(ply_data_grid)):
                # if ply_data_grid[i_idx][3] < 1:  # 0 is surface, -1 is occluded
                # if ply_data_grid[i_idx][3] != STSDF_EMPTY:  # 1 is surface, -1 is occluded, 0 is free
                #     ply_data.append(ply_data_grid[i_idx])
                if ply_data_grid[i_idx][3] == 1:  # 1 is surface
                    ply_data.append(ply_data_grid[i_idx])
        if encoding == 'TSDF':
            for i_idx in range(len(ply_data_grid)):
                # if ply_data_grid[i_idx][3] < -0.0011:  # 0 is surface, <0 is occluded, > 0 is free
                if ply_data_grid[i_idx][3] != TSDF_EMPTY:
                    ply_data.append(ply_data_grid[i_idx])
        if encoding == 'EMPTY':
            for i_idx in range(len(ply_data_grid)):
                # if ply_data_grid[i_idx][3] > 0:  # 0 is free
                if ply_data_grid[i_idx][3] != 0:  # 0 is free
                    ply_data.append(ply_data_grid[i_idx])
        if len(ply_data) == 0:
            print('From _depth_voxel2ply(): NO valid data. {}'.format(ply_filename))
            return
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property int label\n' \
                   'end_header' % len(ply_data)
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d", header=ply_head, comments='')
        print('Saved-->{}'.format(ply_filename))

    @classmethod
    def _gn_voxel2ply(cls, voxel_val, ply_filename, encoding='GN'):
        # ---- get size
        size = voxel_val.shape
        # print('size', size)
        _x, _y, _z = cls._get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        # g = np.zeros(voxel_val.size, dtype=np.float32)
        # b = np.zeros(voxel_val.size, dtype=np.float32)
        gb = np.zeros(voxel_val.size, dtype=np.uint8)
        gn_voxel_val = (voxel_val * 255/1.8).astype(np.uint8)  # max(gn) = 1.8
        ply_data_grid = zip(_x, _y, _z, gn_voxel_val.flatten(), gb, gb)
        ply_data = []
        if encoding == 'GN':
            for i_idx in range(len(ply_data_grid)):
                if ply_data_grid[i_idx][3] > 0:  # 0 is free
                    ply_data.append(ply_data_grid[i_idx])

        if len(ply_data) == 0:
            print('From _depth_voxel2ply(): NO valid data. {}'.format(ply_filename))
            return
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')
        print('Saved-->{}'.format(ply_filename))

    @classmethod
    def labeled_voxel2ply(cls, vox_labeled, ply_filename):  # TODO: 该函数存在内存泄露风险
        """Save labeled voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
           注意 vox_labeled.shape: (W, H, D)
        """  #
        # ---- Check data type, numpy ndarray
        if type(vox_labeled) is not np.ndarray:
            raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(vox_labeled)))
        # ---- Check data validation
        if np.amax(vox_labeled) == 0:
            print('Oops! All voxel is labeled empty.')
            return
        # ---- get size
        size = vox_labeled.shape
        # ---- Convert to list
        vox_labeled = vox_labeled.flatten()
        # ---- Get X Y Z
        _x, _y, _z = cls._get_xyz(size)
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        # ---- Get R G B
        vox_labeled[vox_labeled == 255] = 0  # empty
        # vox_labeled[vox_labeled == 255] = 12  # ignore
        _rgb = colorMap[vox_labeled[:]]
        # ---- Get X Y Z R G B
        xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])
        # xyz_rgb = zip(_z, _y, _x, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # 将X轴和Z轴交换，用于meshlab显示
        # ---- Get ply data without empty voxel
        xyz_rgb = np.array(xyz_rgb)
        ply_data = xyz_rgb[np.where(vox_labeled > 0)]

        if len(ply_data) == 0:
            raise Exception("Oops!  That was no valid ply data.")
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        # ---- Save ply data to disk
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')  # It takes 20s
        del vox_labeled, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head
        # print('Saved-->{}'.format(ply_filename))

    @staticmethod
    def _downsample_label(label, downscale=4):
        r"""downsample the labeled data, new version takes only about 0.6s on CPU
        Shape:
            label, (240, 144, 240)
            label_downscale, if downsample==4, then (60, 36, 60)
        """
        if downscale == 1:
            return label
        ds = int(downscale)
        small_size = (int(voxel_SIZE[0] / ds), int(voxel_SIZE[1] / ds), int(voxel_SIZE[2] / ds))  # small size
        label_downscale = np.zeros(small_size, dtype=np.uint8)
        empty_t = 0.95 * ds * ds * ds  # threshold
        s01 = small_size[0] * small_size[1]
        label_i = np.zeros((ds, ds, ds), dtype=np.int32)
        for i in range(small_size[0]*small_size[1]*small_size[2]):
            z = i / s01
            y = (i - z * s01) / small_size[0]
            x = i - z * s01 - y * small_size[0]
            x = int(x)
            y = int(y)
            z = int(z)
            label_i[:, :, :] = label[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            label_bin = label_i.flatten()
            # zero_count = np.array(np.where(np.logical_or(label_i < 0.001, label_i > 254))).size
            # zero_count = np.array(np.where(np.logical_or(label_bin == 0, label_bin == 255))).size
            # if zero_count < empty_t:
            #     label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
            #     label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
            zero_count_0 = np.array(np.where(label_bin == 0)).size
            zero_count_255 = np.array(np.where(label_bin == 255)).size
            zero_count = zero_count_0 + zero_count_255

            if zero_count > empty_t:
                label_downscale[x, y, z] = 0 if zero_count_0 > zero_count_255 else 255
            else:
                # label_i_s = label_bin[np.nonzero(label_bin)]  # get the none empty class labels
                label_i_s = label_bin[np.where(np.logical_and(label_bin > 0, label_bin < 255))]
                label_downscale[x, y, z] = np.argmax(np.bincount(label_i_s))
        return label_downscale

    @staticmethod  # 0 for empty, 1 for surface, -1 for occ
    def _downsample_stsdf(stsdf, downscale=4):
        r"""
        Shape:
            stsdf, (240, 144, 240)
            stsdf_downscale, (60, 36, 60)
        """
        if downscale == 1:
            return stsdf
        STSDF_EMPTY = 0
        STSDF_SURFACE = 1
        STSDF_OCCLUD = -1
        ds = downscale
        small_size = (np.int32(stsdf.shape[0] / ds), np.int32(stsdf.shape[1] / ds), np.int32(stsdf.shape[2] / ds))
        t = ds * ds * ds * 0.05  # t = 0
        stsdf_downscale = np.zeros(small_size, dtype=np.float32)  # init 0 for empty
        # stsdf_downscale = np.full(small_size, -2.0, dtype=np.float32)  # init -2 for occ
        s01 = small_size[0] * small_size[1]
        stsdf_i = np.ones((ds, ds, ds), dtype=np.float32)
        for i in range(small_size[0] * small_size[1] * small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            stsdf_i[:, :, :] = stsdf[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            stsdf_bin = stsdf_i.flatten()
            none_empty_count = np.array(np.where(stsdf_bin != STSDF_EMPTY)).size  # 0 for empty
            # if none_empty_count > t:
            #     # surface_count  = np.array(np.where(stsdf_bin == 1)).size
            #     # occluded_count = np.array(np.where(stsdf_bin == -2)).size
            #     surface_count  = np.array(np.where(stsdf_bin > 0)).size
            #     occluded_count = np.array(np.where(stsdf_bin < 0)).size
            #     stsdf_downscale[x, y, z] = STSDF_SURFACE if surface_count >= occluded_count else STSDF_OCCLUD
            if none_empty_count > t:
                surface_count = np.array(np.where(stsdf_bin > 0)).size
                # occluded_count = np.array(np.where(stsdf_bin < 0)).size
                # stsdf_downscale[x, y, z] = STSDF_SURFACE if surface_count > t else STSDF_OCCLUD
                # stsdf_downscale[x, y, z] = STSDF_SURFACE if (surface_count > t or surface_count >= occluded_count) else STSDF_OCCLUD
                stsdf_downscale[x, y, z] = STSDF_SURFACE if surface_count > 2 else STSDF_OCCLUD  # 至少3个点才视为surface
            # else:
            #     stsdf_downscale[x, y, z] = 0  # 0 is empty, default is 0
        return stsdf_downscale

    @staticmethod
    def _downsample_tsdf(tsdf, downscale=4):  # 仅在Get None empty　时会用到
        r"""
        Shape:
            tsdf, (240, 144, 240)
            tsdf_downscale, (60, 36, 60), (stsdf.shape[0]/4, stsdf.shape[1]/4, stsdf.shape[2]/4)
        """
        if downscale == 1:
            return tsdf
        # TSDF_EMPTY = np.float32(0.001)
        # TSDF_SURFACE: 1, sign >= 0
        # TSDF_OCCLUD: sign < 0  np.float32(-0.001)
        ds = downscale
        small_size = (int(tsdf.shape[0] / ds), int(tsdf.shape[1] / ds), int(tsdf.shape[2] / ds))
        tsdf_downscale = np.ones(small_size, dtype=np.float32) * np.float32(0.001)  # init 0.001 for empty
        s01 = small_size[0] * small_size[1]
        tsdf_sr = np.ones((ds, ds, ds), dtype=np.float32)  # search region
        for i in range(small_size[0] * small_size[1] * small_size[2]):
            z = int(i / s01)
            y = int((i - z * s01) / small_size[0])
            x = int(i - z * s01 - y * small_size[0])
            tsdf_sr[:, :, :] = tsdf[x * ds:(x + 1) * ds, y * ds:(y + 1) * ds, z * ds:(z + 1) * ds]
            tsdf_bin = tsdf_sr.flatten()
            # none_empty_count = np.array(np.where(tsdf_bin != TSDF_EMPTY)).size
            none_empty_count = np.array(np.where(np.logical_or(tsdf_bin <= 0, tsdf_bin == 1))).size
            if none_empty_count > 0:
                # surface_count  = np.array(np.where(stsdf_bin == 1)).size
                # occluded_count = np.array(np.where(stsdf_bin == -2)).size
                # surface_count = np.array(np.where(tsdf_bin > 0)).size  # 这个存在问题
                surface_count  = np.array(np.where(tsdf_bin == 1)).size
                # occluded_count = np.array(np.where(tsdf_bin < 0)).size
                # tsdf_downscale[x, y, z] = 0 if surface_count > occluded_count else np.float32(-0.001)
                tsdf_downscale[x, y, z] = 1 if surface_count > 2 else np.float32(-0.001)  # 1 or 0 ?
            # else:
            #     tsdf_downscale[x, y, z] = empty  # TODO 不应该将所有值均设为0.001
        return tsdf_downscale

    @staticmethod
    def get_nonempty(voxels, encoding):  # Get none empty from depth voxels
        data = np.zeros(voxels.shape, dtype=np.float32)  # init 0 for empty
        # if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
        #     data[voxels == 1] = 1
        #     return data
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels != 0] = 1
            surface = np.array(np.where(voxels == 1))  # surface=1
        elif encoding == 'TSDF':
            data[np.where(np.logical_or(voxels <= 0, voxels == 1))] = 1
            surface = np.array(np.where(voxels == 1))  # surface
            # surface = np.array(np.where(np.logical_and(voxels > 0, voxels != np.float32(0.001))))  # surface
        else:
            raise Exception("Encoding error: {} is not validate".format(encoding))

        min_idx = np.amin(surface, axis=1)
        max_idx = np.amax(surface, axis=1)
        # print('min_idx, max_idx', min_idx, max_idx)
        # data[:a], data[a]不包含在内, data[b:], data[b]包含在内
        # min_idx = min_idx
        max_idx = max_idx + 1
        # 本该扩大一圈就够了，但由于GT标注的不是很精确，故在高分辨率情况下，多加大一圈
        # min_idx = min_idx - 1
        # max_idx = max_idx + 2
        min_idx[min_idx < 0] = 0
        max_idx[0] = min(voxels.shape[0], max_idx[0])
        max_idx[1] = min(voxels.shape[1], max_idx[1])
        max_idx[2] = min(voxels.shape[2], max_idx[2])
        data[:min_idx[0], :, :] = 0  # data[:a], data[a]不包含在内
        data[:, :min_idx[1], :] = 0
        data[:, :, :min_idx[2]] = 0
        data[max_idx[0]:, :, :] = 0  # data[b:], data[b]包含在内
        data[:, max_idx[1]:, :] = 0
        data[:, :, max_idx[2]:] = 0
        return data

    @staticmethod
    def get_nonempty2(voxels, target, encoding):  # Get none empty from depth voxels
        data = np.ones(voxels.shape, dtype=np.float32)  # init 1 for none empty
        data[target == 255] = 0
        if encoding == 'STSDF':  # surface, empty, occulted: 1, 0, -1
            data[voxels == 0] = 0
        elif encoding == 'TSDF':
            # --0
            # data[voxels == np.float32(0.001)] = 0
            # --1
            # data[voxels > 0] = 0
            # --2
            # data[voxels >= np.float32(0.001)] = 0
            # --3
            data[voxels >= np.float32(0.001)] = 0
            data[voxels == 1] = 1

        return data

    @classmethod
    def rgb_voxel2ply(cls, rgb_vox, ply_filename):
        """Save rgb voxels to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
        input:
            rgb_vox, shape (240, 144, 240, 3)
        """
        # ---- [0,1]-->[0,255]
        rgb_vox = rgb_vox * 255
        # ---- get size
        size = rgb_vox.shape  # (240, 144, 240, 3)
        rgb_vox = rgb_vox.reshape(-1, 3)
        # ---- Get X Y Z
        _x, _y, _z = cls._get_xyz(size[0:3])
        _x = _x.flatten()
        _y = _y.flatten()
        _z = _z.flatten()
        # ---- Get X Y Z R G B
        xyz_rgb = zip(_x, _y, _z, rgb_vox[:, 0], rgb_vox[:, 1], rgb_vox[:, 2])
        # ---- Get ply data without empty voxel
        # -- Method 1
        # ply_data = []
        # for i_idx in range(rgb_vox.shape[0]):
        #     # if rgb_vox[i_idx, :].any():  # 0 is empty
        #     if rgb_vox[i_idx, 0] > 0 or rgb_vox[i_idx, 1] > 0 or rgb_vox[i_idx, 2] > 0:  # 0 is empty
        #         ply_data.append(xyz_rgb[i_idx])
        # -- Method 2, twice faster than method1
        xyz_rgb = np.array(xyz_rgb)
        ply_data = xyz_rgb[np.where(rgb_vox.any(axis=1))]

        if len(ply_data) == 0:
            raise Exception("Oops!  That was no valid ply data.")
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        # ---- Save ply data to disk
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')
        del rgb_vox, _x, _y, _z, xyz_rgb, ply_data, ply_head
        # print('Saved-->{}'.format(ply_filename))

    @staticmethod
    def _get_point_cloud(rgb, depth, filename):
        """ Get color point cloud from RGB-D. rgb (H, W, 3), depth(H, W)
        """  #
        # rgb = cls._read_rgb(filename + '.jpg', 480, 640)  # (H, W, 3)     # ---- Read RGB jpg
        # depth = cls._read_depth(filename+'.png', 480, 640)  # (480, 640)  # ---- Read depth

        # ---- Get point in camera coordinate
        gx, gy = np.meshgrid(range(W), range(H))
        point_cam = np.zeros((H, W, 6), dtype=np.float32)
        point_cam[:, :, 0] = (gx - cam_K[0][2]) * depth / cam_K[0][0]  # x
        point_cam[:, :, 1] = (gy - cam_K[1][2]) * depth / cam_K[1][1]  # y
        point_cam[:, :, 2] = depth  # z, in meter
        point_cam[:, :, 3] = rgb[:, :, 0]
        point_cam[:, :, 4] = rgb[:, :, 1]
        point_cam[:, :, 5] = rgb[:, :, 2]
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % depth.size
        ply_filename = filename + '_RGBD2CPL.ply'
        ply_data = point_cam.reshape((-1, 6))
        np.savetxt(ply_filename, ply_data, fmt="%.2f %.2f %.2f %d %d %d", header=ply_head, comments='')
        print('Saved-->{}'.format(ply_filename))

    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------
    # ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ---------- ----------


    @classmethod
    def gn2hist(cls, gn, target, minv=0, maxv=3, minz=0):
        gn_gray = gn
        # gn_gray = (gn - minv) / (maxv - minv) * 10
        # gn_gray = gn_gray.astype(np.int32)
        print('gn2hist', type(gn_gray), type(gn_gray[0][0][0]))

        unique, counts = numpy.unique(gn_gray, return_counts=True)
        print(dict(zip(unique, counts)))

    @classmethod
    def gn2ply(cls, gn, target, ply_filename, minv=0, maxv=3, minz=0):
        """minz 为截面的Z坐标，小于Z的点不再显示，故形成截面示意图
           minv=0, maxv=3 为gn的最大值和最小值
        """
        # print('gn', np.amin(gn), np.amax(gn))  # 0, 1
        # gn_gray = (gn - np.amin(gn))/(np.amax(gn) - np.amin(gn)) * 255 + 0

        # gn_gray = gn * 255
        gn_gray = (gn - minv) / (maxv - minv) * 255# + 0
        # print(gn.shape)
        # w, h, d = 240, 144, 240
        w, h, d = gn_gray.shape
        # print(w, h, d)
        # rgb = np.zeros(gn.shape+(3,), dtype=np.int8)
        # print(rgb.shape)
        # rgb[target == 0] = (0, 0, 0)
        ply_data = []

        for i in range(w):
            for j in range(h):
                for k in range(minz, d):
                    if target[i, j, k] != 0 and target[i, j, k] != 255:
                        r = GetR(gn_gray[i, j, k])
                        g = GetG(gn_gray[i, j, k])
                        b = GetB(gn_gray[i, j, k])
                        # rgb[i, j, k] = (r, g, b)
                        ply_data.append((i, j, k, r, g, b))

        # 去除target中为empty的部分

        # ---- [0,1]-->[0,255]
        # rgb_vox = rgb_vox * 255
        # # ---- get size
        # size = rgb_vox.shape  # (240, 144, 240, 3)
        # rgb_vox = rgb_vox.reshape(-1, 3)
        # # ---- Get X Y Z
        # _x, _y, _z = cls._get_xyz(size[0:3])
        # _x = _x.flatten()
        # _y = _y.flatten()
        # _z = _z.flatten()
        # # ---- Get X Y Z R G B
        # xyz_rgb = zip(_x, _y, _z, rgb_vox[:, 0], rgb_vox[:, 1], rgb_vox[:, 2])
        # # ---- Get ply data without empty voxel
        # # -- Method 1
        # # ply_data = []
        # # for i_idx in range(rgb_vox.shape[0]):
        # #     # if rgb_vox[i_idx, :].any():  # 0 is empty
        # #     if rgb_vox[i_idx, 0] > 0 or rgb_vox[i_idx, 1] > 0 or rgb_vox[i_idx, 2] > 0:  # 0 is empty
        # #         ply_data.append(xyz_rgb[i_idx])
        # # -- Method 2, twice faster than method1
        # xyz_rgb = np.array(xyz_rgb)
        # ply_data = xyz_rgb[np.where(rgb_vox.any(axis=1))]

        if len(ply_data) == 0:
            raise Exception("Oops!  That was no valid ply data.")
        ply_head = 'ply\n' \
                   'format ascii 1.0\n' \
                   'element vertex %d\n' \
                   'property float x\n' \
                   'property float y\n' \
                   'property float z\n' \
                   'property uchar red\n' \
                   'property uchar green\n' \
                   'property uchar blue\n' \
                   'end_header' % len(ply_data)
        # ---- Save ply data to disk
        np.savetxt(ply_filename, ply_data, fmt="%d %d %d %d %d %d", header=ply_head, comments='')
        # del rgb_vox, _x, _y, _z, xyz_rgb, ply_data, ply_head
        print('Saved-->{}'.format(ply_filename))



def one_hot_embedding(labels, num_classes):
    r"""Embedding labels to one-hot form.
    Args:
        labels: (LongTensor) class labels, sized [N,**].
        num_classes: (int) number of classes.
    Returns:
        (numpy) encoded labels, sized [N,#classes].
    """
    shape = labels.shape
    y = np.eye(num_classes)  # [C,C]
    y_one_hot = y[labels.flatten()]  # [N,C]
    y_one_hot = y_one_hot.reshape(shape + (num_classes,))
    return y_one_hot
# ---------------------------------------------------------------------------------------------------------------

# def GetR(gray):
#     return gray
#
# def GetG(gray):
#     return 125
#
# def GetB(gray):
#     return 255 - gray


def GetR(gray):
    if gray < 127:
        return 0
    elif gray > 191:
        return 255
    else:
        return (gray - 127) * 4 - 1


def GetG(gray):
    if gray < 64:
        return 4 * gray
    elif gray > 191:
        return 256 - (gray - 191) * 4
    else:
        return 255


def GetB(gray):
    if gray < 64:
        return 255
    elif gray > 127:
        return 0
    else:
        return 256 - (gray - 63) * 4

if __name__ == '__main__':
    # ---- Data loader
    data_dir = '/home/jie/fastDATA/NYUCADvalidate40-lr-ply--depth2stream-nonempty2-color'
    data_loader = torch.utils.data.DataLoader(
        dataset=SuncgDataset(data_dir, 'TRAIN', encoding='BINARY', downsample=1),
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    import datetime
    time1 = datetime.datetime.now()

    # ---- LR
    for step, (voxel_rgbd, target_lr, _filename) in enumerate(data_loader):
        # rgb_vox.T, target_vox.T, one_hot_target_vox.T, _name + '.png'
        print('step:', step, voxel_rgbd.shape, len(_filename))
    # print(datetime.datetime.now() - time1)
