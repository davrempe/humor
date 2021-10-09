import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import os.path as osp
import glob, time, copy, pickle, json, math

from torch.utils.data import Dataset, DataLoader

from fitting.fitting_utils import read_keypoints, load_planercnn_res

import numpy as np
import torch
import cv2

DEFAULT_GROUND = [0.0, -1.0, 0.0, -0.5]

class RGBVideoDataset(Dataset):

    def __init__(self, joints2d_path,
                       cam_mat,
                       seq_len=None,
                       overlap_len=None,
                       img_path=None,
                       load_img=False,
                       masks_path=None,
                       mask_joints=False,
                       planercnn_path=None,
                       video_name='rgb_video'
                 ):
        '''
        Creates a dataset based on a single RGB video.

        - joints2d_path : path to saved OpenPose keypoints for the video
        - cam_mat : 3x3 camera intrinsics
        - seq_len : If not none, the maximum number of frames in a subsequence, will split the video into subsequences based on this. If none, the dataset contains a single sequence of the whole video.
        - overlap_len : the minimum number of frames to overlap each subsequence if splitting the video.
        - img_path : path to directory of video frames
        - load_img : if True, will load and return the video frames as part of the data.
        - masks_path : path to person segmentation masks
        - mask_joints: if True, masks the returned 2D joints using the person segmentation masks (i.e. drops any occluded joints)
        - planercnn_path : path to planercnn results on a single frame of the video. If given, uses this ground plane in the returned
                            data rather than the default.
        '''
        super(RGBVideoDataset, self).__init__()

        self.joints2d_path = joints2d_path
        self.cam_mat = cam_mat
        self.seq_len = seq_len
        self.overlap_len = overlap_len
        self.img_path = img_path
        self.load_img = load_img
        self.masks_path = masks_path
        self.mask_joints = mask_joints
        self.planercnn_path = planercnn_path
        self.video_name = video_name

        # load data paths 
        self.data_dict, self.seq_intervals = self.load_data()
        self.data_len = len(self.data_dict['joints2d'])
        print('RGB dataset contains %d sub-sequences...' % (self.data_len))

    def load_data(self):
        '''
        Loads in the full dataset, except for image frames which are loaded on the fly if desired.
        '''
        
        # get length of sequence based on files in self.joints2d_path and split into even sequences.
        keyp_paths = sorted(glob.glob(osp.join(self.joints2d_path, '*_keypoints.json')))
        frame_names = ['_'.join(f.split('/')[-1].split('_')[:-1]) for f in keyp_paths]
        num_frames = len(keyp_paths)
        print('Found video with %d frames...' % (num_frames))

        seq_intervals = []
        if self.seq_len is not None and self.overlap_len is not None:
            num_seqs = math.ceil((num_frames - self.overlap_len) / (self.seq_len - self.overlap_len))
            r = self.seq_len*num_seqs - self.overlap_len*(num_seqs-1) - num_frames # number of extra frames we cover
            extra_o = r // (num_seqs - 1) # we increase the overlap to avoid these as much as possible
            self.overlap_len = self.overlap_len + extra_o

            new_cov = self.seq_len*num_seqs - self.overlap_len*(num_seqs-1) # now compute how many frames are still left to account for
            r = new_cov - num_frames

            # create intervals
            cur_s = 0
            cur_e = cur_s + self.seq_len
            for int_idx in range(num_seqs):
                seq_intervals.append((cur_s, cur_e))
                cur_overlap = self.overlap_len
                if int_idx < r:
                    cur_overlap += 1 # update to account for final remainder
                cur_s += (self.seq_len - cur_overlap)
                cur_e = cur_s + self.seq_len

            print('Splitting into subsequences of length %d frames overlapping by %d...' % (self.seq_len, self.overlap_len))
        else:
            print('Not splitting the video...')
            num_seqs = 1
            self.seq_len = num_frames
            seq_intervals = [(0, self.seq_len)]

        #
        # first load in entire video then split
        #

        # intrinsics
        cam_mat = self.cam_mat

        # path to image frames
        img_paths = None
        if self.img_path is not None:
            img_paths = [osp.join(self.img_path, img_fn)
                            for img_fn in os.listdir(self.img_path)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
            img_paths = sorted(img_paths)

        # path to masks
        mask_paths = None
        if self.masks_path is not None:
            mask_paths = [os.path.join(self.masks_path, f + '.png') for f in frame_names]

        # floor plane
        floor_plane = None
        if self.planercnn_path is not None:
            floor_plane = load_planercnn_res(self.planercnn_path)
        else:
            floor_plane = np.array(DEFAULT_GROUND)

        # get data for each subsequence
        data_out = {
            'img_paths' : [],
            'mask_paths' : [],
            'cam_matx' : [],
            'joints2d' : [],
            'floor_plane' : [],
            'names' : []
        }
        for seq_idx in range(num_seqs):
            sidx, eidx = seq_intervals[seq_idx]

            data_out['cam_matx'].append(cam_mat)

            keyp_frames = [read_keypoints(f) for f in keyp_paths[sidx:eidx]]
            joint2d_data = np.stack(keyp_frames, axis=0) # T x J x 3 (x,y,conf)
            data_out['joints2d'].append(joint2d_data)

            data_out['floor_plane'].append(floor_plane)

            data_out['names'].append(self.video_name + '_' + '%04d' % (seq_idx))

            if img_paths is not None:
                data_out['img_paths'].append(img_paths[sidx:eidx])
            if mask_paths is not None:
                data_out['mask_paths'].append(mask_paths[sidx:eidx])

        return data_out, seq_intervals

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        obs_data = dict()
        gt_data = dict()

        #
        # 2D keypoints
        #
        joint2d_data = self.data_dict['joints2d'][idx]
        obs_data['joints2d'] = torch.Tensor(joint2d_data)

        # person mask
        if self.mask_joints:
            cur_mask_paths = self.data_dict['mask_paths'][idx]
            obs_data['mask_paths'] = cur_mask_paths

            for t, mask_file in enumerate(cur_mask_paths):
                # load in mask
                vis_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                imh, imw = vis_mask.shape
                # mask out invisible joints (give confidence 0)
                uvs = np.round(joint2d_data[t, :, :2]).astype(int)
                uvs[:,0][uvs[:,0] >= imw] = (imw-1)
                uvs[:,1][uvs[:,1] >= imh] = (imh-1)
                occluded_mask_idx = vis_mask[uvs[:, 1], uvs[:, 0]] != 0
                joint2d_data[t, :, :][occluded_mask_idx] = 0.0
        
        # images
        if len(self.data_dict['img_paths']) > 0:
            cur_img_paths = self.data_dict['img_paths'][idx]
            obs_data['img_paths'] = cur_img_paths
            if self.load_img:
                # load images
                img_list = []
                for img_path in cur_img_paths:
                    # print(img_path)
                    img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
                    img_list.append(img)
                img_out = torch.Tensor(np.stack(img_list, axis=0))
                # print(img_out.size())
                obs_data['RGB'] = img_out

                # import matplotlib.pyplot as plt
                # for t in range(self.seq_len):
                #     fig = plt.figure()
                #     plt.imshow(img_list[t])
                #     plt.scatter(joint2d_data[t, :, 0], joint2d_data[t, :, 1])
                #     ax = plt.gca()
                #     plt.show()
                #     plt.close(fig)
                
                # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                # for img in img_list:
                #     while True:
                #         cv2.imshow('frame', img)
                #         key = cv2.waitKey(30)
                #         if key == 27:
                #             break

        # floor plane
        obs_data['floor_plane'] = self.data_dict['floor_plane'][idx]
        # intrinsics
        gt_data['cam_matx'] = torch.Tensor(self.data_dict['cam_matx'][idx])
        # meta-data
        gt_data['name'] = self.data_dict['names'][idx]

        # the frames used in this subsequence
        obs_data['seq_interval'] = torch.Tensor(list(self.seq_intervals[idx])).to(torch.int)

        return obs_data, gt_data