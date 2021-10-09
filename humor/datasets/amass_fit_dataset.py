import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import numpy as np
from torch.utils.data import Dataset
import torch

from datasets.amass_utils import CONTACT_INDS
from utils.transforms import rotation_matrix_to_angle_axis
from body_model.body_model import BodyModel
from datasets.amass_discrete_dataset import AmassDiscreteDataset
from body_model.utils import SMPLH_PATH, SMPL_JOINTS

class AMASSFitDataset(Dataset):
    '''
    Wrapper around AmassDiscreteDataset to return observed and GT data as expected and add desired noise.
    '''

    def __init__(self, data_path,
                       seq_len=60,
                       return_joints=True,
                       return_verts=True,
                       return_points=True,
                       noise_std=0.0,
                       make_partial=False,
                       partial_height=0.75,
                       drop_middle=False,
                       num_samp_pts=512,
                       root_only=False,
                       split_by='dataset',
                       custom_split=None):

        self.seq_len = seq_len # global seq returns + 1
        self.return_joints = return_joints
        self.return_verts = return_verts
        self.return_points = return_points
        self.num_samp_pts = num_samp_pts
        self.noise_std = noise_std
        self.make_partial = make_partial
        self.partial_height = partial_height
        self.drop_middle = drop_middle
        self.root_only = root_only
        split_str = 'test'
        if split_by == 'dataset' and custom_split is not None:
            split_str = 'custom'
        self.amass_dataset = AmassDiscreteDataset(split=split_str,
                                                  data_paths=[data_path],
                                                  split_by=split_by,
                                                  sample_num_frames=seq_len - 1, # global seq returns + 1
                                                  step_frames_in=1,
                                                  step_frames_out=1,
                                                  data_rot_rep='aa',
                                                  data_return_config='all',
                                                  only_global=True,
                                                  custom_split=custom_split)
        
        if self.return_points:
            # need to have SMPL model to sample on surface
            self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
            male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
            female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
            self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=self.seq_len).to(self.device)
            self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=self.seq_len).to(self.device)

    def __len__(self):
        return self.amass_dataset.__len__()

    def __getitem__(self, idx):
        global_data, meta = self.amass_dataset.__getitem__(idx)
        
        # create the ground truth data dictionary
        gt_dict = dict()
        for k, v in global_data.items():
            gt_key = '_'.join(k.split('_')[1:])
            T = v.size(0)
            if gt_key == 'root_orient' or gt_key == 'pose_body':
                # convert mat to aa rots
                v = rotation_matrix_to_angle_axis(v.reshape((-1, 3, 3))).reshape((T, -1))
            gt_dict[gt_key] = v
        gt_dict['betas'] = meta['betas']
        gt_dict['gender'] = meta['gender']
        gt_dict['name'] = meta['path'].replace('/', '_')[:-4]

        # create clean observations
        observed_dict = dict()
        if self.return_joints:
            # 3d joint positions
            observed_dict['joints3d'] = gt_dict['joints'].clone().detach()
            if self.root_only:
                for k, v in SMPL_JOINTS.items():
                    if k not in ['leftArm', 'rightArm', 'head', 'neck', 'hips']:
                        observed_dict['joints3d'][:,v,:] = float('inf') # everything but root is not observed
        if self.return_verts:
            # 3d vertex positions (of specific chosen keypoint vertices)
            observed_dict['verts3d'] = gt_dict['verts'].clone().detach()
        if self.return_points:
            import trimesh
            # forward pass of SMPL
            cur_bm = self.male_bm if meta['gender'] == 'male' else self.female_bm
            body_gt = cur_bm(pose_body=gt_dict['pose_body'].to(self.device), 
                             pose_hand=None, 
                             betas=gt_dict['betas'].to(self.device),
                             root_orient=gt_dict['root_orient'].to(self.device),
                             trans=gt_dict['trans'].to(self.device))
            # sample points on the surface
            nv = body_gt.v.size(1)
            gt_dict['points'] = body_gt.v
            points_list = []
            for t in range(self.seq_len):
                verts = body_gt.v[t].cpu().detach().numpy()
                faces = body_gt.f.cpu().detach().numpy()
                body_mesh = trimesh.Trimesh(vertices=verts,
                                            faces=faces,
                                            process=False)
                pts_t = trimesh.sample.sample_surface(body_mesh, self.num_samp_pts)[0]
                points_list.append(pts_t)
            points = torch.Tensor(np.stack(points_list, axis=0))
            observed_dict['points3d'] = points

        # add gaussian noise
        if self.noise_std > 0.0:
            for k in observed_dict.keys():
                observed_dict[k] += torch.randn_like(observed_dict[k])*self.noise_std

        if self.make_partial:
            # if z below a certain threshold make occluded
            for k, v in observed_dict.items():
                if k == 'joints3d' and self.root_only:
                    continue
                occluded_mask = observed_dict[k][:,:,2:3] < self.partial_height #0.95 #0.75
                occluded_mask = occluded_mask.expand_as(observed_dict[k])
                observed_dict[k][occluded_mask] = float('inf')

                if k == 'points3d':
                    vis_mask = torch.logical_not(occluded_mask)
                    cur_points_seq = observed_dict[k]
                    for t in range(self.seq_len):
                        vis_pts = cur_points_seq[t][vis_mask[t]]
                        vis_pts = vis_pts.reshape((-1, 3))
                        vis_pts = resize_points(vis_pts, self.num_samp_pts)
                        observed_dict[k][t] = vis_pts

        if self.drop_middle:
            for k, v in observed_dict.items():
                sidx = self.seq_len // 3
                eidx = sidx + (self.seq_len // 3)
                observed_dict[k][sidx:eidx] = float('inf')

        if 'contacts' in gt_dict:
            gt_contacts = gt_dict['contacts']
            full_contacts = torch.zeros((gt_contacts.size(0), len(SMPL_JOINTS))).to(gt_contacts)
            full_contacts[:,CONTACT_INDS] = gt_contacts
            gt_dict['contacts'] = full_contacts

        return observed_dict, gt_dict