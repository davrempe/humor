import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import os.path as osp
import glob, time, copy, pickle, json

from torch.utils.data import Dataset, DataLoader

from utils.transforms import batch_rodrigues, rotation_matrix_to_angle_axis

from fitting.fitting_utils import read_keypoints, resize_points, OP_FLIP_MAP, load_planercnn_res

import numpy as np
import torch
import cv2

TRIM_EDGES = 90 # number of frames to cut off beginning and end of qualitative

QUAL_FPS = 30
QUANT_FPS = 5

QUANT_TRAIN = ['vicon']
QUANT_TEST = ['vicon']
QUANT_SPLITS = [QUANT_TRAIN, QUANT_TEST]

QUAL_TRAIN = ['BasementSittingBooth', 'MPH16', 'N0SittingBooth', 'N3Office', 'MPH112', 'MPH1Library', 'N0Sofa', 'N3OpenArea', 
               'MPH11', 'MPH8', 'N3Library', 'Werkraum']
QUAL_TEST = ['N3Office', 'N0Sofa', 'N3Library', 'MPH1Library']
QUAL_SPLITS = [QUAL_TRAIN, QUAL_TEST]

# these are the only SMPL parameters we care about
#   rename with our convention
SMPL_NAME_MAP = {
    'transl' : 'trans',
    'beta' : 'betas',
    'body_pose' : 'pose_body',
    'global_orient' : 'root_orient',
    'betas' : 'betas' # sometimes it's named differently in qualitative data
}
SMPL_SIZES = {
    'trans' : 3,
    'betas' : 10,
    'pose_body' : 63,
    'root_orient' : 3
}

FEMALE_SUBJ_IDS = [162, 3452, 159, 3403]
DEPTH_SCALE = 1e-3
IMG_WIDTH, IMG_HEIGHT = 1920, 1080

def read_fitting_seq(fitting_paths, return_valid=False):
    '''
    Reads in a sequence of PROX/PROXD SMPL fits and concats into single data dict as
    torch tensor.
    - return_valid : if true, returns a corresponding bool list of which fits are not dummy (i.e. there was actually a fitting loaded in)
    '''
    val_names = [SMPL_NAME_MAP[k] for k in SMPL_NAME_MAP.keys()]
    fit_dict = {val_name : [] for val_name in val_names}
    valid_list = []
    for fpath in fitting_paths:
        if not os.path.exists(fpath):
            print('No PROX results exist for %s, filling with dummy zeros!' % (fpath))
            for k, v in SMPL_SIZES.items():
                fit_dict[k].append(torch.zeros((v), dtype=torch.float).reshape((1, -1)))
            valid_list.append(False)
        else:
            with open(fpath, 'rb') as f:
                param = pickle.load(f, encoding='latin1')

            cur_valid = True
            for key in param.keys():
                if key in SMPL_NAME_MAP.keys():
                    cur_valid = cur_valid and (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(param[key])))).item() == 0)
                    if cur_valid:
                        fit_dict[SMPL_NAME_MAP[key]].append(torch.Tensor(param[key]).reshape((1, -1)))
                    else:
                        fit_dict[SMPL_NAME_MAP[key]].append(torch.zeros((SMPL_SIZES[SMPL_NAME_MAP[key]]), dtype=torch.float).reshape((1, -1)))
            if not cur_valid:
                print('PROX results nan for %s, filling with dummy zeros!' % (fpath))


            valid_list.append(cur_valid)
    
    fit_dict = {k : torch.cat(v, dim=0) for k, v in fit_dict.items()}
    if return_valid:
        return fit_dict, valid_list
    else:
        return fit_dict

class ProxDataset(Dataset):
    '''
    NOTE: support for quantitative dataset has not been thoroughly tested.
    '''

    def __init__(self, root_path,
                       quant=False, # quant or qual dataset
                       split='train',
                       seq_len=10, # split the data into sequences of this length
                       load_depth=False,
                       max_pts=4096, # max number of points to return from depth image
                       load_img=False,
                       load_scene_mesh=False,
                       estimate_floor_plane=False, # if true, estimates the ground plane from the scene mesh and returns it
                       load_floor_plane=False, # if true, loads the PlaneRCNN floor plane from the dataset and uses this
                       mask_color=True, # whether to load mask from RGB (NN) or from Kinect
                       mask_joints=False, # whether to apply the mask to 2d joints so that occluded joints are (0,0,0)
                       return_mask=False, # whether to return the mask or not
                       recording=None, # if given, loads only this single recording
                       recording_subseq_idx=-1, # if given, loads only this single subsequence of a specified recording
                       return_fitting=True, # if true, loads SMPL params fit from MOSH (quant) or PROXD (qual) as "ground truth"
                       flip=True # reflects images and masks about y axis, MUST be true so that GT fitting matches data and given scene geometry
                 ):
        super(ProxDataset, self).__init__()

        self.root_path = root_path
        self.quant = quant
        data_dir = 'quantitative' if self.quant else 'qualitative'
        self.data_dir = os.path.join(self.root_path, data_dir)
        self.seq_len = seq_len
        self.load_depth = load_depth
        self.max_pts = max_pts
        self.load_img = load_img
        self.load_scene_mesh = load_scene_mesh
        self.estimate_floor_plane = estimate_floor_plane
        self.load_floor_plane = load_floor_plane
        self.mask_color = mask_color
        self.mask_joints = mask_joints
        self.return_mask = return_mask
        self.split = split
        self.recording = recording
        self.recording_subseq_idx = recording_subseq_idx
        if self.recording is None and self.recording_subseq_idx > 0:
            print('Ignoring subseq_idx since recording not specified...')
            self.recording_subseq_idx = -1
        self.return_fitting = return_fitting
        self.flip = flip

        if self.mask_joints and not self.mask_color:
            print('Must be using color mask in order to mask joints (since detected in RGB)! Will NOT mask joints...')
            self.mask_joints = False

        # data roots
        self.rec_root = os.path.join(self.data_dir, 'recordings')
        self.calib_dir = os.path.join(self.data_dir, 'calibration')
        self.cam2world_root = os.path.join(self.data_dir, 'cam2world')
        self.fitting_root = os.path.join(self.data_dir, 'fittings/mosh') if self.quant else \
                       os.path.join(self.data_dir, 'PROXD')
        self.keypoints_root = os.path.join(self.data_dir, 'keypoints')
        self.planes_root = os.path.join(self.data_dir, 'planes')
        self.scenes_root = os.path.join(self.data_dir, 'scenes')
        
        data_splits = QUANT_SPLITS if self.quant else QUAL_SPLITS
        self.split_scenes = data_splits[0] if self.split == 'train' else data_splits[1]

        # load (img) data paths 
        self.img_paths, self.subseq_inds = self.load_data()
        self.data_len = len(self.img_paths)
        print('This split contains %d sub-sequences...' % (self.data_len))

    def load_data(self):
        # camera intrinsics are the same for all sequences/scenes
        self.projection = Projection(self.calib_dir)
        
        # get the sequences we want
        recording_list = []
        if self.recording is not None:
            rec_path = os.path.join(self.rec_root, self.recording)
            if os.path.exists(rec_path):
                recording_list = [rec_path]
            else:
                print('Could not find specified recording at %s!' % (rec_path))
        else:
            all_rec_dirs = [os.path.join(self.rec_root, f) for f in sorted(os.listdir(self.rec_root)) if f[0] != '.']
            all_rec_dirs = [f for f in all_rec_dirs if os.path.isdir(f)]
            recording_list = [f for f in all_rec_dirs if f.split('/')[-1].split('_')[0] in self.split_scenes]
            
        recording_names = [f.split('/')[-1] for f in recording_list]

        print('Found %d recordings in this split...' % (len(recording_names)))
        print('Splitting into subsequences of length %d frames...' % (self.seq_len))

        # split each recording into sequences and record information for loading data
        img_path_list = []
        subseq_idx_list = [] # sub index into the recording
        for rec_path, rec_name in zip(recording_list, recording_names):
            img_folder = osp.join(rec_path, 'Color')
            img_paths = [osp.join(img_folder, img_fn)
                            for img_fn in os.listdir(img_folder)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
            img_paths = sorted(img_paths)

            # print(img_paths)
            cur_rec_len = len(img_paths)
            # cut off edges of qualitative data to avoid static parts
            if not self.quant and (cur_rec_len - (2*TRIM_EDGES)) >= self.seq_len:
                img_paths = img_paths[TRIM_EDGES:-TRIM_EDGES]
                cur_rec_len = len(img_paths)

            if len(img_paths) < self.seq_len:
                continue

            # split into max number of sequences of desired length
            num_seqs = cur_rec_len // self.seq_len

            if self.recording_subseq_idx > -1:
                sidx = self.recording_subseq_idx*self.seq_len
                eidx = sidx + self.seq_len
                seq_paths = img_paths[sidx:eidx]
                img_path_list.append(seq_paths)
                subseq_idx_list.append(self.recording_subseq_idx)
            else:
                for i in range(num_seqs):
                    sidx = i*self.seq_len
                    eidx = sidx + self.seq_len
                    seq_paths = img_paths[sidx:eidx]
                    img_path_list.append(seq_paths)
                    subseq_idx_list.append(i)
        return img_path_list, subseq_idx_list

    def get_data_paths_from_img(self, img_paths):
        # return paths for all other data modalities from the img_paths for a sequence
        rec_path = '/'.join(img_paths[0].split('/')[:-2])
        rec_name = rec_path.split('/')[-1]
        frame_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in img_paths]

        keyp_folder = osp.join(self.keypoints_root, rec_name)
        depth_folder = os.path.join(rec_path, 'Depth')
        mask_folder = os.path.join(rec_path, 'BodyIndex')
        mask_color_folder = os.path.join(rec_path, 'BodyIndexColor')
        fitting_folder = osp.join(self.fitting_root, rec_name, 'results')

        keyp_paths = [osp.join(keyp_folder, f + '_keypoints.json') for f in frame_names]
        depth_paths = [osp.join(depth_folder, f + '.png') for f in frame_names]
        mask_paths = [osp.join(mask_folder, f + '.png') for f in frame_names]
        mask_color_paths = [osp.join(mask_color_folder, f + '.png') for f in frame_names]
        fitting_paths = [osp.join(fitting_folder, f, '000.pkl') for f in frame_names]

        return keyp_paths, depth_paths, mask_paths, mask_color_paths, fitting_paths

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        obs_data = dict()
        gt_data = dict()

        cur_img_paths = self.img_paths[idx]
        obs_data['img_paths'] = cur_img_paths
        rec_name = cur_img_paths[0].split('/')[-3]
        # get other data paths
        keyp_paths, depth_paths, mask_paths, mask_color_paths, fitting_paths = self.get_data_paths_from_img(cur_img_paths)
        obs_data['mask_paths'] = mask_color_paths if self.mask_color else mask_paths

        # load desired data
        # load mask or mask color depending on settings
        mask_list = []
        if self.mask_color:
            for mask_file in mask_color_paths:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask_list.append(mask)
        else:
            for mask_file in mask_paths:
                mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)[1]
                mask_list.append(mask)
        if self.flip:
            mask_list = [cv2.flip(mask, 1) for mask in mask_list]
        if self.return_mask:
            mask_seq_out = torch.Tensor(np.stack(mask_list, axis=0))
            obs_data['mask'] = mask_seq_out

        # always load keypoints
        keyp_frames = [read_keypoints(f) for f in keyp_paths]
        joint2d_data = np.stack(keyp_frames, axis=0) # T x J x 3 (x,y,conf)
        if (not self.quant and self.flip) or (self.quant and not self.flip):
            # quant keypoints are already flipped (so need to unflip if necessary)
            joint2d_data = joint2d_data[:, OP_FLIP_MAP, :] # reflect labeling
            joint2d_data[:, :, 0] = IMG_WIDTH - joint2d_data[:, :, 0] # visually reflect about y

        # mask out invisible joints if desired (give confidence 0)
        if self.mask_joints and self.mask_color:
            for t, mask in enumerate(mask_list):
                uvs = np.round(joint2d_data[t, :, :2]).astype(int)
                uvs[:,0][uvs[:,0] >= IMG_WIDTH] = (IMG_WIDTH-1)
                uvs[:,1][uvs[:,1] >= IMG_HEIGHT] = (IMG_HEIGHT-1)
                occluded_mask_idx = mask[uvs[:, 1], uvs[:, 0]] != 0
                joint2d_data[t, :, :][occluded_mask_idx] = 0.0

        obs_data['joints2d'] = torch.Tensor(joint2d_data)

        # load images
        if self.load_img:
            img_list = []
            for img_path in cur_img_paths:
                img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
                if self.flip:
                    img = cv2.flip(img, 1)
                img_list.append(img)

            img_out = torch.Tensor(np.stack(img_list, axis=0))
            # print(img_out.size())
            obs_data['RGB'] = img_out
            
        if self.quant:
            vicon2scene = np.eye(4)
            with open(os.path.join(self.data_dir, 'vicon2scene.json'), 'r') as f:
                vicon2scene = np.array(json.load(f))
            gt_data['vicon2scene'] = torch.Tensor(vicon2scene)

        # load GT fitting SMPL params
        if self.return_fitting:
            fitting_data = read_fitting_seq(fitting_paths)
            for k, v in fitting_data.items():
                gt_data[k] = v

        # load depth and backproject to point cloud if desired, mask to be only visible points
        if self.load_depth:
            # load in each depth image
            depth_img_list = []
            for depth_path in depth_paths:
                depth_im = cv2.imread(depth_path, flags=-1).astype(float)
                depth_im = depth_im / 8.
                depth_im = depth_im * DEPTH_SCALE
                if self.flip:
                    depth_im = cv2.flip(depth_im, 1)
                depth_img_list.append(depth_im)
            
            # mask so only visible points stay
            points_list = []
            for depth_img, mask in zip(depth_img_list, mask_list):
                scan_dict = self.projection.create_scan(mask, depth_img, mask_on_color=self.mask_color, coord='color') # want points in RGB camera system
                cur_points = scan_dict['points']
                if cur_points.shape[0] == 0:
                    print('No points in depth map!')
                    # the person is completely invisible, just copy over previous frame if possible
                    if len(points_list) > 0:
                        print('Copying previous frame...')
                        cur_points = points_list[-1]
                    else:
                        print('filling zeros...')
                        cur_points = np.zeros((self.max_pts, 3))
                else:
                    cur_points = resize_points(cur_points, self.max_pts)
                points_list.append(cur_points)
            points = np.stack(points_list, axis=0)
            obs_data['points3d'] = torch.Tensor(points)

        # load camera extrinsics and intrinsics (return only RGB since already projected points)
        scene_name = rec_name.split('_')[0]
        cam2world_path = os.path.join(self.cam2world_root, scene_name + '.json')
        cam2world = np.eye(4)
        with open(cam2world_path, 'r') as f:
            cam2world = np.array(json.load(f))
        gt_data['cam2world'] = torch.Tensor(cam2world)
        gt_data['cam_matx'] = torch.Tensor(self.projection.color_cam['camera_mtx'])

        if self.load_floor_plane:
            # load in parameters and masks
            planes_path = os.path.join(self.planes_root, scene_name)
            floor_plane = load_planercnn_res(planes_path)
            obs_data['floor_plane'] = floor_plane

        if self.load_scene_mesh or self.estimate_floor_plane:
            import trimesh
            scene_mesh = trimesh.load(os.path.join(self.scenes_root, scene_name + '.ply'))
            scene_verts = np.array(scene_mesh.vertices)
            scene_faces = np.array(scene_mesh.faces)
            if self.load_scene_mesh:
                gt_data['scene'] = (scene_verts, scene_faces)
            if self.estimate_floor_plane:
                from sklearn.cluster import DBSCAN
                point_heights = scene_verts[:,2]
                neg_mask = point_heights < 0
                point_heights = point_heights[neg_mask]
                neg_points = scene_verts[neg_mask, :]
                samp_n_pts = 10000
                if neg_points.shape[0] > samp_n_pts:
                    samp_inds = np.random.choice(np.arange(neg_points.shape[0]), size=samp_n_pts, replace=False)
                    neg_points = neg_points[samp_inds]
                    point_heights = point_heights[samp_inds]
                # cluster point heights
                clustering = DBSCAN(eps=0.005, min_samples=100).fit(point_heights.reshape((-1, 1)))
                num_clusters = np.unique(clustering.labels_).shape[0] #clustering.components_.shape[0]
                max_cluster_size = -float('inf')
                max_clust_idx = -1
                cluster_sizes = []
                for clust_idx in range(num_clusters):
                    cur_clust_size = np.sum(clustering.labels_ == clust_idx)
                    cluster_sizes.append(cur_clust_size)
                
                sort_inds = np.argsort(np.array(cluster_sizes))[::-1]
                max_clust_label = sort_inds[0]
                max_clust_mean = np.mean(neg_points[clustering.labels_ == max_clust_label], axis=0)
                next_clust_label = sort_inds[1]
                next_clust_mean = np.mean(neg_points[clustering.labels_ == next_clust_label], axis=0)
                max_clust_idx = max_clust_label if max_clust_mean[2] <= next_clust_mean[2] else next_clust_label

                floor_points = neg_points[clustering.labels_ == max_clust_idx]

                # fit the floor to these points
                from sklearn.linear_model import RANSACRegressor
                reg = RANSACRegressor(random_state=0).fit(floor_points[:,:2], floor_points[:,2])
                est = reg.estimator_
                plane_normal = np.array([-est.coef_[0], -est.coef_[1], 1.0])
                norm_mag = np.linalg.norm(plane_normal)
                plane_normal = plane_normal / norm_mag
                plane_intercept = est.intercept_ / norm_mag
                a, b, c = plane_normal
                d = plane_intercept

                # transform into the camera frame
                cam2world_R = cam2world[:3, :3]
                cam2world_t = cam2world[:3, 3]
                world2cam_R = cam2world_R.T
                world2cam_t = -np.matmul(world2cam_R, cam2world_t)
                new_normal = np.dot(world2cam_R, plane_normal)
                point_on_old = np.array([0.0, 0.0, d / c])
                point_on_new = np.dot(world2cam_R, point_on_old) + world2cam_t
                new_intercept = np.dot(new_normal, point_on_new)
                a, b, c = new_normal
                d = new_intercept
                floor_plane = np.array([a, b, c, d])

                obs_data['floor_plane'] = floor_plane
        
        # other meta-data
        cur_name = rec_name + '_' + '%04d' % (self.subseq_inds[idx])
        cur_subj_id = cur_name.split('_')[1]
        gender = 'female' if int(cur_subj_id) in FEMALE_SUBJ_IDS else 'male'
        gt_data['name'] = cur_name
        gt_data['gender'] = gender

        return obs_data, gt_data


#
# Adapted from https://github.com/mohamedhassanmus/prox/blob/master/prox/projection_utils.py
# Please see their license for usage restrictions.
#
class Projection():
    def __init__(self, calib_dir):
        with open(osp.join(calib_dir, 'IR.json'), 'r') as f:
            self.depth_cam = json.load(f)
        with open(osp.join(calib_dir, 'Color.json'), 'r') as f:
            self.color_cam = json.load(f)

    def row(self, A):
        return A.reshape((1, -1))
    def col(self, A):
        return A.reshape((-1, 1))

    def unproject_depth_image(self, depth_image, cam):
        us = np.arange(depth_image.size) % depth_image.shape[1]
        vs = np.arange(depth_image.size) // depth_image.shape[1]
        ds = depth_image.ravel()
        uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)
        #unproject
        xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()),
                                                      np.asarray(cam['camera_mtx']), np.asarray(cam['k']))
        xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), self.col(uvd[:, 2])))
        xyz_camera_space[:, :2] *= self.col(xyz_camera_space[:, 2])  # scale x,y by z
        other_answer = xyz_camera_space - self.row(np.asarray(cam['view_mtx'])[:, 3])  # translate
        xyz = other_answer.dot(np.asarray(cam['view_mtx'])[:, :3])  # rotate

        return xyz.reshape((depth_image.shape[0], depth_image.shape[1], -1))

    def projectPoints(self, v, cam):
        v = v.reshape((-1,3)).copy()
        return cv2.projectPoints(v, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']), np.asarray(cam['k']))[0].squeeze()

    def create_scan(self, mask, depth_im, color_im=None, mask_on_color=False, coord='color', TH=1e-2, default_color=[1.00, 0.75, 0.80]):
        if not mask_on_color:
            depth_im[mask != 0] = 0
        if depth_im.size == 0:
            return {'v': []}

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        colors = np.tile(default_color, [points.shape[0], 1])

        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        if mask_on_color:
            valid_mask_idx = valid_idx.copy()
            valid_mask_idx[valid_mask_idx == True] = mask[uvs[valid_idx == True][:, 1], uvs[valid_idx == True][:, 0]] == 0
            uvs = uvs[valid_mask_idx == True]
            points = points[valid_mask_idx]
            colors = np.tile(default_color, [points.shape[0], 1])
            # colors = colors[valid_mask_idx]
            valid_idx = valid_mask_idx
            if color_im is not None:
                colors[:, :3] = color_im[uvs[:, 1], uvs[:, 0]] / 255.0
        else:
            uvs = uvs[valid_idx == True]
            if color_im is not None:
                colors[valid_idx == True,:3] = color_im[uvs[:, 1], uvs[:, 0]]/255.0

        if coord == 'color':
            # Transform to color camera coord
            T = np.concatenate([np.asarray(self.color_cam['view_mtx']), np.array([0, 0, 0, 1]).reshape(1, -1)])
            stacked = np.column_stack((points, np.ones(len(points)) ))
            points = np.dot(T, stacked.T).T[:, :3]
            points = np.ascontiguousarray(points)
        ind = points[:, 2] > TH
        return {'points':points[ind], 'colors':colors[ind]}


    def align_color2depth(self, depth_im, color_im, interpolate=True):
        (w_d, h_d) = (512, 424)
        if interpolate:
            # fill depth holes to avoid black spots in aligned rgb image
            zero_mask = np.array(depth_im == 0.).ravel()
            depth_im_flat = depth_im.ravel()
            depth_im_flat[zero_mask] = np.interp(np.flatnonzero(zero_mask), np.flatnonzero(~zero_mask),
                                                 depth_im_flat[~zero_mask])
            depth_im = depth_im_flat.reshape(depth_im.shape)

        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]
        aligned_color = np.zeros((h_d, w_d, 3)).astype(color_im.dtype)
        aligned_color[valid_idx.reshape(h_d, w_d)] = color_im[uvs[:, 1], uvs[:, 0]]

        return aligned_color

    def align_depth2color(self, depth_im, depth_raw):
        (w_rgb, h_rgb) = (1920, 1080)
        (w_d, h_d) = (512, 424)
        points = self.unproject_depth_image(depth_im, self.depth_cam).reshape(-1, 3)
        uvs = self.projectPoints(points, self.color_cam)
        uvs = np.round(uvs).astype(int)
        valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)
        valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
        valid_idx = np.logical_and(valid_x, valid_y)
        uvs = uvs[valid_idx == True]

        aligned_depth = np.zeros((h_rgb, w_rgb)).astype('uint16')
        aligned_depth[uvs[:, 1], uvs[:, 0]] = depth_raw[valid_idx.reshape(h_d, w_d)]

        return aligned_depth