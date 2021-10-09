import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import os.path as osp
import glob, time, copy, pickle, json

from torch.utils.data import Dataset, DataLoader

from fitting.fitting_utils import read_keypoints, load_planercnn_res

import numpy as np
import torch
import cv2

SPLIT = ['Scene04', 'Scene05', 'Scene07', 'Scene10', 'Scene11', 'Scene12', 'Scene13', 'Scene14']
SCENE_MAP = {'Scene04' : 'lobby19-3',
             'Scene05' : 'lobby18-1',
             'Scene07' : 'lobby15',
             'Scene10' : 'lobby22-1-tog',
             'Scene11' : 'livingroom00',
             'Scene12' : 'office1-1-tog-lcrnet',
             'Scene13' : 'library3-tog',
             'Scene14' : 'garden1' }

# GT 3d joints do not align with images.
QUANT_BLACKLIST = ['Scene04', 'Scene12']
# for some reason 3d joint annotation is off by one frame
SHIFT_LIST = ['Scene05']

# which scene objects contact the ground, used for floor fit.
GROUND_CTC_PARTS = {'Scene05' : ['00_couch_seat'],
                    'Scene07' : ['00_couch_seat', '01_couch_seat'],
                    'Scene10' : ['00_couch_seat', '01_couch_seat', '03_couch_seat'],
                    'Scene11' : ['00_couch_seat', '01_couch_seat', '01_couch_seat-1', '02_couch_seat'],
                    'Scene13' : ['04_couch_seat', '05_chair_leg', '05_chair_leg-1', '05_chair_leg-2', '05_chair_leg-3', 
                                 '07_chair_leg', '07_chair_leg-1', '07_chair_leg-2', '07_chair_leg-3',
                                 '08_chair_leg', '08_chair_leg-1', '08_chair_leg-2', '08_chair_leg-3',
                                 '09_chair_leg', '09_chair_leg-1', '09_chair_leg-2', '09_chair_leg-3'],
                    'Scene14' : ['01_chair_leg', '01_chair_leg-1', '01_chair_leg-3', '01_chair_leg-4',
                                 '02_chair_leg', '02_chair_leg-1', '02_chair_leg-3', '02_chair_leg-4',
                                 '03_chair_leg', '03_chair_leg-1', '03_chair_leg-3', '03_chair_leg-4',
                                 '04_table_leg', '04_table_leg-1', '04_table_leg-2', '04_table_leg-3',
                                 '00_couch_leg', '00_couch_leg-1', '00_couch_leg-3', '00_couch_leg-4']}


IMG_WIDTH, IMG_HEIGHT = 1920, 1080

class iMapperDataset(Dataset):

    def __init__(self, root_path,
                       seq_len=10, # split the data into sequences of this length
                       load_img=False,
                       load_floor_plane=False, # if true, loads the PlaneRCNN floor plane from the dataset and uses this
                       scene=None, # if given, loads only this scene
                       scene_subseq_idx=-1, # if given, loads only this single subsequence of a specified recording
                       load_gt_floor=False, # if true, uses object locations to get GT floor
                       load_scene_mesh=False,
                       mask_joints=False # if true, masks 2d joints based on person mask
                 ):
        super(iMapperDataset, self).__init__()

        self.data_dir = root_path
        self.seq_len = seq_len
        self.load_img = load_img
        self.load_floor_plane = load_floor_plane
        self.load_gt_floor = load_gt_floor
        self.load_scene_mesh = load_scene_mesh
        self.scene = scene
        self.scene_subseq_idx = scene_subseq_idx
        self.mask_joints = mask_joints
        if self.scene is None and self.scene_subseq_idx > 0:
            print('Ignoring subseq_idx since scene is not specified...')
            self.scene_subseq_idx = -1
        
        self.split_scenes = SPLIT

        # load (img) data paths 
        self.data_dict = self.load_data()
        self.data_len = len(self.data_dict['img_paths'])
        print('This split contains %d sub-sequences...' % (self.data_len))

    def load_data(self):
        '''
        Loads in the full dataset, except for image frames which are loaded on the fly.
        '''
        
        # get the sequences we want
        scene_list = []
        if self.scene is not None:
            scene_path = os.path.join(self.data_dir, self.scene)
            if os.path.exists(scene_path):
                scene_list = [scene_path]
            else:
                print('Could not find specified scene at %s!' % (scene_path))
        else:
            all_scene_dirs = [os.path.join(self.data_dir, scene_name) for scene_name in self.split_scenes]
            scene_list = [f for f in all_scene_dirs if os.path.exists(f) and os.path.isdir(f)]
            
        scene_names = [f.split('/')[-1] for f in scene_list]

        print('Found %d scenes...' % (len(scene_names)))
        print('Splitting into subsequences of length %d frames...' % (self.seq_len))

        # split each scene into sequences and record information for loading data
        data_out = {
            'img_paths' : [],
            'mask_paths' : [],
            'cam_matx' : [],
            'joints2d' : [],
            'joints3d' : [],
            'occlusions' : [],
            'floor_plane' : [],
            'gt_floor_plane' : [],
            'scene_mesh' : [],
            'names' : []
        }
        for scene_path, scene_name in zip(scene_list, scene_names):
            # first load in data for full sequence
            scene_data = dict()

            #
            # path to image frames
            #
            img_folder = osp.join(scene_path, 'raw_frames')
            img_paths = [osp.join(img_folder, img_fn)
                            for img_fn in os.listdir(img_folder)
                            if img_fn.endswith('.png') or
                            img_fn.endswith('.jpg') and
                            not img_fn.startswith('.')]
            img_paths = sorted(img_paths)
            frame_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in img_paths]

            mask_folder = osp.join(scene_path, 'masks')
            mask_paths = [os.path.join(mask_folder, f + '.png') for f in frame_names]

            cur_seq_len = len(img_paths)
            if len(img_paths) < self.seq_len:
                continue

            if scene_name in QUANT_BLACKLIST:
                continue

            scene_data['img_paths'] = img_paths
            scene_data['mask_paths'] = mask_paths

            #
            # intrinsics
            # 
            intrins_path = osp.join(scene_path, 'intrinsics.json')
            with open(intrins_path, 'r') as f:
                intrins_data = json.load(f)
            cam_mat = np.array(intrins_data)


            #
            # 2d observed keypoints (OpenPose)
            #
            keyp_folder = osp.join(scene_path, 'op_keypoints') 
            keyp_paths = [osp.join(keyp_folder, f + '_keypoints.json') for f in frame_names]
            keyp_frames = [read_keypoints(f) for f in keyp_paths]
            joint2d_data = np.stack(keyp_frames, axis=0) # T x J x 3 (x,y,conf)
            nobs_frames = joint2d_data.shape[0]

            scene_data['joints2d'] = joint2d_data

            #
            # scene GT info which gives 3d joints, occlusions for each joint, and ground plane
            #
            scene_info_path = osp.join(scene_path, 'gt/skel_%s_GT.json' % (SCENE_MAP[scene_name]))
            with open(scene_info_path, 'r') as f:
                scene_info = json.load(f)

            # first joints 3d
            joints3d = []
            for k, v in sorted(scene_info['3d'].items()):
                frame_id = int(k)
                pose = np.zeros(shape=(len(v[list(v.keys())[0]]), len(v)),
                                dtype=np.float32)
                for joint, pos in v.items():
                    pose[:, int(joint)] = pos
                joints3d.append(pose.T)

            joints3d = np.stack(joints3d, axis=0)
            ngt_frames = joints3d.shape[0]
            ngt_joints = joints3d.shape[1]

            # confidence in 3d joint annotations (some are marked 0.0 and need to be thrown out)
            conf3d = []
            for frame_id, v in enumerate(scene_info['confidence']['values']):
                cur_conf = np.zeros((ngt_joints, 1), dtype=np.float32)
                for joint, jconf in v.items():
                    cur_conf[int(joint)] = float(jconf)
                conf3d.append(cur_conf)
            conf3d = np.stack(conf3d, axis=0)
            # now use it to mask out 3d joints (inf means not observed)
            conf3d[conf3d == 0.0] = float('inf')
            joints3d = joints3d * conf3d

            if scene_name in SHIFT_LIST:
                joints3d_shift = np.ones_like(joints3d)*float('inf')
                joints3d_shift[:-1] = joints3d[1:]
                joints3d = joints3d_shift

            occlusion_mask = np.zeros((ngt_frames, ngt_joints), dtype=np.int)
            for k, v in scene_info['occluded'].items():
                frame_id = int(k) - 1 
                occlusion_mask[frame_id] = v

            floor_trans = np.array(scene_info['ground'])
            floor_rot = np.array(scene_info['ground_rot'])

            # align GT subsampled data to full 30 hz sampling rate observed data
            joints3d_aligned = np.ones((nobs_frames, ngt_joints, 3), dtype=np.float)*float('inf')
            occlusions_aligned = np.ones((nobs_frames, ngt_joints), dtype=np.float)*float('inf')
            sub_data_list = [joints3d, occlusion_mask]
            aligned_data_list = [joints3d_aligned, occlusions_aligned]
            for cur_arr, aligned_arr in zip(sub_data_list, aligned_data_list):
                # first few entries are non-regular due to video->frame conversion at different fps
                aligned_arr[0] = cur_arr[0]
                aligned_arr[2] = cur_arr[1]
                aligned_arr[3] = cur_arr[2]
                aligned_arr[5] = cur_arr[3]
                # remaining entries are regularly spaced as expected
                cur_aligned_idx = 5
                for sub_idx in range(4, ngt_frames):
                    cur_aligned_idx += 3
                    aligned_arr[cur_aligned_idx] = cur_arr[sub_idx]

            scene_data['joints3d'] = aligned_data_list[0]
            scene_data['occlusions'] = aligned_data_list[1]

            # 
            # PlaneRCNN floor plane
            # 
            if self.load_floor_plane:
                # load in parameters and masks
                planes_path = os.path.join(scene_path, 'planes')
                floor_plane = load_planercnn_res(planes_path)
            else:
                floor_plane = np.array([0.0, -1.0, 0.0, -floor_trans[1, 3]])

            #
            # Estimate GT from objects
            #
            gt_floor_plane = None
            scene_mesh = None
            if self.load_gt_floor or self.load_scene_mesh:
                scene_obj = scene_info['scenelets']['obbs']
                scene_json = [os.path.join(scene_path, 'gt', f.replace('.obj', '.json')) for f in scene_obj]
                part_names = [f.split('/')[-1].split('.')[0] for f in scene_json]

                scene_verts = []
                scene_faces = []
                ground_ctc_verts = []
                ground_ctc_faces = []
                floor_fit_pts = []
                for part_idx, part_json_path in enumerate(scene_json):
                    obb, data_obb = Obb.load(part_json_path,
                                                return_data=True)

                    part_verts = np.array(obb.corners_3d())
                    part_fs = np.array(obb.face_ids())

                    scene_verts.append(part_verts)
                    scene_faces.append(part_fs + part_idx*8) # there are 8 vertices in each part

                    if part_names[part_idx] in GROUND_CTC_PARTS[scene_name]:
                        ground_ctc_verts.append(part_verts)
                        ground_ctc_faces.append(part_fs + len(ground_ctc_faces)*8)

                        ground_verts = obb.corners_3d_lower()
                        floor_fit_pts.append(ground_verts)

                scene_verts = np.concatenate(ground_ctc_verts, axis=0)
                scene_faces = np.concatenate(ground_ctc_faces, axis=0)
                if self.load_scene_mesh:
                    scene_mesh = (scene_verts, scene_faces)

                floor_fit_pts = np.concatenate(floor_fit_pts, axis=0)
                if self.load_gt_floor:
                    # fit floor plane
                    from sklearn.linear_model import RANSACRegressor, LinearRegression
                    est = LinearRegression().fit(floor_fit_pts[:,[0,2]], floor_fit_pts[:,1])
                    plane_normal = np.array([-est.coef_[0], 1.0, -est.coef_[1]])
                    norm_mag = np.linalg.norm(plane_normal)
                    plane_normal = plane_normal / norm_mag
                    plane_intercept = est.intercept_ / norm_mag
                    if plane_normal[1] > 0.0:
                        # should always face up 
                        plane_normal *= -1.0
                        plane_intercept *= -1.0
                    a, b, c = plane_normal
                    d = plane_intercept
                    gt_floor_plane = np.array([a, b, c, d])

            #
            # then split into subsequences
            #

            # split into max number of sequences of desired length
            num_seqs = cur_seq_len // self.seq_len

            if self.scene_subseq_idx > -1:
                sidx = self.scene_subseq_idx*self.seq_len
                eidx = sidx + self.seq_len
                for k, v in scene_data.items():
                    data_out[k].append(scene_data[k][sidx:eidx])
                data_out['names'].append(scene_name + '_' + '%04d' % (self.scene_subseq_idx))
                # same for all subsequences
                data_out['cam_matx'].append(cam_mat)
                data_out['floor_plane'].append(floor_plane)
                if self.load_gt_floor:
                    data_out['gt_floor_plane'].append(gt_floor_plane)
                if self.load_scene_mesh:
                    data_out['scene_mesh'].append(scene_mesh)
            else:
                for i in range(num_seqs):
                    sidx = i*self.seq_len
                    eidx = sidx + self.seq_len
                    for k, v in scene_data.items():
                        data_out[k].append(scene_data[k][sidx:eidx])
                    data_out['names'].append(scene_name + '_' + '%04d' % (i))
                    data_out['cam_matx'].append(cam_mat)
                    data_out['floor_plane'].append(floor_plane)
                    if self.load_gt_floor:
                        data_out['gt_floor_plane'].append(gt_floor_plane)
                    if self.load_scene_mesh:
                        data_out['scene_mesh'].append(scene_mesh)

        return data_out

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
        cur_mask_paths = self.data_dict['mask_paths'][idx]
        obs_data['mask_paths'] = cur_mask_paths

        if self.mask_joints:
            for t, mask_file in enumerate(cur_mask_paths):
                # load in mask
                vis_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                # mask out invisible joints (give confidence 0)
                uvs = np.round(joint2d_data[t, :, :2]).astype(int)
                uvs[:,0][uvs[:,0] >= IMG_WIDTH] = (IMG_WIDTH-1)
                uvs[:,1][uvs[:,1] >= IMG_HEIGHT] = (IMG_HEIGHT-1)
                occluded_mask_idx = vis_mask[uvs[:, 1], uvs[:, 0]] != 0
                joint2d_data[t, :, :][occluded_mask_idx] = 0.0
        

        #
        # images
        #
        cur_img_paths = self.data_dict['img_paths'][idx]
        obs_data['img_paths'] = cur_img_paths
        # load images
        if self.load_img:
            img_list = []
            for img_path in cur_img_paths:
                # print(img_path)
                img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
                img_list.append(img)
            img_out = torch.Tensor(np.stack(img_list, axis=0))
            # print(img_out.size())
            obs_data['RGB'] = img_out

        #
        # joints3d
        # 
        joint3d_data = self.data_dict['joints3d'][idx]
        gt_data['joints3d'] = torch.Tensor(joint3d_data)

        #
        # occlusions
        #
        gt_data['occlusions'] = torch.Tensor(self.data_dict['occlusions'][idx])

        #
        # floor plane
        #
        # if self.load_floor_plane:
        obs_data['floor_plane'] = self.data_dict['floor_plane'][idx]
        if self.load_gt_floor:
            gt_data['floor_plane'] = self.data_dict['gt_floor_plane'][idx]

        # intrinsics
        gt_data['cam_matx'] = torch.Tensor(self.data_dict['cam_matx'][idx])
        
        # meta-data
        gt_data['name'] = self.data_dict['names'][idx]

        # scene mesh
        if self.load_scene_mesh:
            gt_data['scene'] = self.data_dict['scene_mesh'][idx]

        return obs_data, gt_data


#
# Code adapted from iMapper (https://github.com/amonszpart/iMapper).
# Please refer to their license for usage restrictions.
#
class Obb(object):
    __corners = [(-1, -1, -1), (-1, -1, 1),
                 (-1, 1, 1), (-1, 1, -1),
                 (1, 1, -1), (1, -1, -1),
                 (1, -1, 1), (1, 1, 1)]
    __face_ids = [(0, 3, 1), (3, 2, 1),
                  (0, 1, 5), (1, 6, 5),
                  (4, 5, 6), (4, 6, 7),
                  (3, 4, 2), (4, 7, 2),
                  (4, 3, 5), (3, 0, 5),
                  (6, 2, 7), (6, 1, 2)]

    def __init__(self, centroid=None, axes=None, scales=None):
        self._centroid = np.asarray(centroid, dtype=np.float32).reshape((3, 1)) \
            if centroid is not None \
            else np.zeros(shape=(3, 1), dtype=np.float32)
        self._axes = axes if axes is not None \
            else np.identity(3, dtype=np.float32)  # axes in cols
        self._scales = np.asarray(scales, dtype=np.float32).reshape((3, 1)) \
            if scales is not None \
            else np.ones(shape=(3, 1), dtype=np.float32)
        """Full side length"""
        self._corners_3d = None
        self._faces_3d = None

    @classmethod
    def corners(cls):
        return cls.__corners

    @classmethod
    def face_ids(cls):
        return cls.__face_ids

    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, vec):
        self._centroid = np.asarray(vec, dtype=np.float32).reshape((3, 1))
        self._corners_3d = None

    @property
    def axes(self):
        return self._axes

    @axes.setter
    def axes(self, axes):
        assert axes.shape == (3, 3), \
            "Not allowed shape: %s" % axes.shape.__repr__()
        self._axes = axes
        self._corners_3d = None

    def axes_scaled(self):
        return np.matmul(np.diag(self.scales.flatten()), self.axes)

    @property
    def scales(self):
        """Full side lengths, not half-axis lengths."""
        return self._scales

    @scales.setter
    def scales(self, scales):
        self._scales = np.asarray(scales).reshape((3, 1))
        self._corners_3d = None

    def axis(self, axis_id):
        assert self._scales.shape == (3, 1), \
            "[SceneObj::axis] wrong _scales shape: %s" % \
            self._scales.shape.__repr__()
        assert self._axes.shape == (3, 3), \
            "[SceneObj::axis] wrong _axes shape: %s" % \
            self._axes.shape.__repr__()
        return self._scales[axis_id] * np.squeeze(self._axes[:, axis_id])

    def set_axis(self, axis_id, value):
        length = np.linalg.norm(value)
        self._axes[:, axis_id] = (value / length).astype(np.float32)
        self._scales[axis_id] = np.float32(length)
        assert self._axes.shape == (3, 3), \
            "[sceneObj::set_axis] shape changed: %s" % \
            self._axes.shape.__repr__()

    def corners_3d(self):
        """
        :return: Corners in rows x 3
        """
        if self._corners_3d is not None:
            return self._corners_3d
        else:
            half_axes = np.zeros(shape=(3, 3), dtype="float")
            for a in range(3):
                half_axes[:, a] = self.axis(axis_id=a) / 2.
            corners_3d = \
                np.zeros((len(Obb.corners()), 3), np.float32)
            for row, corner in enumerate(Obb.corners()):
                corners_3d[row, :] = \
                    self.centroid.T \
                    + half_axes[:, 0] * corner[0] \
                    + half_axes[:, 1] * corner[1] \
                    + half_axes[:, 2] * corner[2]
            # print("corners_3d: %s" % corners_3d)
            self._corners_3d = corners_3d
            return corners_3d

    def corners_3d_lower(self, up_axis=(0., -1., 0.)):
        """Returns the 4 points that have smaller y coordinates given up_axis"""
        c3d = self.corners_3d()
        dots = np.dot(c3d, up_axis)
        indices = np.argsort(dots)
        # return c3d[sorted(indices[:4]), :]
        corners_tmp = c3d[sorted(indices[:4]), :]
        for i in range(0, 4):
            i1 = i + 1
            if i1 > 3:
                i1 -= 4
            i2 = i + 2
            if i2 > 3:
                i2 -= 4
            len_diag = np.linalg.norm(corners_tmp[i2, :] - corners_tmp[i, :])
            len_side = np.linalg.norm(corners_tmp[i1, :] - corners_tmp[i, :])
            if len_side > len_diag:
                corners_tmp[i1, :], corners_tmp[i2, :] = \
                    corners_tmp[i2, :], corners_tmp[i1, :].copy()
                # assert corners_tmp[i1] != corners_tmp[i2], \
                #     "Wrong: %s" % corners_tmp
        return corners_tmp

    def faces_3d(self):
        corners_3d = self.corners_3d()
        assert corners_3d.shape[1] == 3, \
            "assumed Nx3: %s" % corners_3d.shape
        faces_3d = np.zeros(shape=(3, 3, len(Obb.face_ids())))
        for face_id, face in enumerate(Obb.face_ids()):
            for d in range(3):
                faces_3d[d, :, face_id] = \
                    corners_3d[face[d], :]
        return faces_3d

    def faces_3d_memoized(self):
        if self._faces_3d is None:
            corners_3d = self.corners_3d()
            self._faces_3d = np.zeros(shape=(3, 3, len(Obb.face_ids())))
            for face_id, face in enumerate(Obb.face_ids()):
                for d in range(3):
                    self._faces_3d[d, :, face_id] = \
                        corners_3d[face[d], :]
        return self._faces_3d

    def rectangles_3d(self):
        corners_3d = self.corners_3d()
        assert corners_3d.shape[1] == 3, \
            "assumed Nx3: %s" % corners_3d.shape
        return np.array(
          [corners_3d[[3, 2, 1, 0], :],
           corners_3d[4:8, :],
           corners_3d[[0, 5, 4, 3], :],
           corners_3d[[6, 1, 2, 7], :],
           corners_3d[[3, 4, 7, 2], :],
           corners_3d[[0, 1, 6, 5], :]])

    def to_obj_string(self, name, vertex_offset=0):
        lines = []
        lines.append("o %s\n" % name)
        corners_3d = self.corners_3d()
        for row in range(corners_3d.shape[0]):
            lines.append("v %f %f %f\n" %
                         (corners_3d[row, 0],
                          corners_3d[row, 1],
                          corners_3d[row, 2]))

        # lines.append("usemtl None\ns off\n")
        # print("vertex_offset: %d" % vertex_offset)
        for face in Obb.face_ids():
            lines.append("f %d %d %d\n" %
                         (vertex_offset+face[0]+1, vertex_offset+face[2]+1,
                          vertex_offset+face[1]+1))
        return "".join(lines)

    def to_json(self, part_id=None):
        d = {'scales':  self.scales.tolist(),
             'centroid': self.centroid.tolist(),
             'axes': self.axes.tolist()}
        if part_id is not None:
            d['part_id'] = part_id
        return d

    @classmethod
    def from_json(cls, data):
        obb = Obb()
        obb.scales = np.asarray(data['scales'])
        obb.centroid = np.asarray(data['centroid'])
        obb.axes = np.asarray(data['axes'])
        return obb

    @classmethod
    def load(cls, path, return_data=False):
        with open(path, 'r') as f_in:
            data = json.load(f_in)
            if return_data:
                return cls.from_json(data), data
            else:
                return cls.from_json(data)
    
    # def __str__(self):
    #     return "Obb(%s, %s, %s)" % (self.centroid.T, self.axes, self.scales.T)

    def __eq__(self, other):
        return np.allclose(self.centroid, other.centroid) \
               and np.allclose(self.axes, other.axes) \
               and np.allclose(self.scales, other.scales)


if __name__=='__main__':
    seq_len = 60
    scene = 'Scene05'
    scene_subseq = 0
    dataset = iMapperDataset( './data/iMapper/i3DB',
                            seq_len=seq_len,
                            load_img=True,
                            load_floor_plane=True,
                            scene=scene,
                            scene_subseq_idx=scene_subseq,
                            load_gt_floor=True,
                            load_scene_mesh=True,
                            mask_joints=False
                            )

    # create loaders
    batch_size = 1
    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False,
                        worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    from viz.utils import viz_smpl_seq, imapper_connections
    from body_model.body_model import BodyModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i, data in enumerate(loader):
        obs_data, gt_data = data

        print(i)
        print(gt_data['name'])

        img_seq = obs_data['RGB'].cpu().numpy()

        joints = gt_data['joints3d'].to(device)
        print(joints.size())

        viz_joints = joints.clone()
        viz_joints[torch.isinf(viz_joints)] = -1.0 # out of sight

        camera_matrix = gt_data['cam_matx'].to(device)
        cam_fx = camera_matrix[:, 0, 0]
        cam_fy = camera_matrix[:, 1, 1]
        cam_cx = camera_matrix[:, 0, 2]
        cam_cy = camera_matrix[:, 1, 2]
        cam_f = torch.stack([cam_fx, cam_fy], dim=1)
        cam_center = torch.stack([cam_cx, cam_cy], dim=1)
        cam_intrins = torch.cat([cam_f, cam_center], dim=1)
        print(cam_intrins)
        camera_intrinsics = []
        for ridx in range(batch_size):
            camera_intrinsics.append(tuple(cam_intrins[ridx].cpu().numpy()))
        print(camera_intrinsics)

        floor_plane = gt_data['floor_plane']
        print(floor_plane)

        viz_scene = None
        # import trimesh
        # scene_verts, scene_faces = gt_data['scene']
        # print(scene_verts)
        # print(scene_faces)
        # scene_mesh = trimesh.Trimesh(vertices=scene_verts[0].cpu().numpy(), faces=scene_faces[0].cpu().numpy(), process=False)
        # viz_scene = [scene_mesh]

        viz_smpl_seq(None, fps=30,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    skel_connections=imapper_connections,
                    ground_plane=None, #floor_plane[0].cpu().numpy(),
                    imw=1920, imh=1080,
                    camera_intrinsics=camera_intrinsics[0],
                    img_seq=img_seq[0],
                    RGBA=True,
                    out_path='./dev_imapper',
                    use_offscreen=True,
                    static_meshes=viz_scene,
                    contacts=None,
                    joints_seq=viz_joints[0])