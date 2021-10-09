import numpy as np

import torch
import torch.nn as nn
import pickle

from smplx import SMPL, SMPLH, SMPLX
from smplx.vertex_ids import vertex_ids
from smplx.utils import Struct

class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    '''

    def __init__(self,
                 bm_path,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=10,
                 use_vtx_selector=False,
                 model_type='smplh'):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.

        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''
        self.use_vtx_selector = use_vtx_selector
        cur_vertex_ids = None
        if self.use_vtx_selector:
            cur_vertex_ids = vertex_ids[model_type]
        data_struct = None
        if '.npz' in bm_path:
            # smplx does not support .npz by default, so have to load in manually
            smpl_dict = np.load(bm_path, encoding='latin1')
            data_struct = Struct(**smpl_dict)
            # print(smpl_dict.files)
            if model_type == 'smplh':
                data_struct.hands_componentsl = np.zeros((0))
                data_struct.hands_componentsr = np.zeros((0))
                data_struct.hands_meanl = np.zeros((15 * 3))
                data_struct.hands_meanr = np.zeros((15 * 3))
                V, D, B = data_struct.shapedirs.shape
                data_struct.shapedirs = np.concatenate([data_struct.shapedirs, np.zeros((V, D, SMPL.SHAPE_SPACE_DIM-B))], axis=-1) # super hacky way to let smplh use 16-size beta
        kwargs = {
                'model_type' : model_type,
                'data_struct' : data_struct,
                'num_betas': num_betas,
                'batch_size' : batch_size,
                'num_expression_coeffs' : num_expressions,
                'vertex_ids' : cur_vertex_ids,
                'use_pca' : False,
                'flat_hand_mean' : True
        }
        assert(model_type in ['smpl', 'smplh', 'smplx'])
        if model_type == 'smpl':
            self.bm = SMPL(bm_path, **kwargs)
            self.num_joints = SMPL.NUM_JOINTS
        elif model_type == 'smplh':
            self.bm = SMPLH(bm_path, **kwargs)
            self.num_joints = SMPLH.NUM_JOINTS
        elif model_type == 'smplx':
            self.bm = SMPLX(bm_path, **kwargs)
            self.num_joints = SMPLX.NUM_JOINTS

        self.model_type = model_type

    def forward(self, root_orient=None, pose_body=None, pose_hand=None, pose_jaw=None, pose_eye=None, betas=None,
                trans=None, dmpls=None, expression=None, return_dict=False, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        assert(dmpls is None)
        out_obj = self.bm(
                betas=betas,
                global_orient=root_orient,
                body_pose=pose_body,
                left_hand_pose=None if pose_hand is None else pose_hand[:,:(SMPLH.NUM_HAND_JOINTS*3)],
                right_hand_pose=None if pose_hand is None else pose_hand[:,(SMPLH.NUM_HAND_JOINTS*3):],
                transl=trans,
                expression=expression,
                jaw_pose=pose_jaw,
                leye_pose=None if pose_eye is None else pose_eye[:,:3],
                reye_pose=None if pose_eye is None else pose_eye[:,3:],
                return_full_pose=True,
                **kwargs
        )

        out = {
            'v' : out_obj.vertices,
            'f' : self.bm.faces_tensor,
            'betas' : out_obj.betas,
            'Jtr' : out_obj.joints,
            'pose_body' : out_obj.body_pose,
            'full_pose' : out_obj.full_pose
        }
        if self.model_type in ['smplh', 'smplx']:
            out['pose_hand'] = torch.cat([out_obj.left_hand_pose, out_obj.right_hand_pose], dim=-1)
        if self.model_type == 'smplx':
            out['pose_jaw'] = out_obj.jaw_pose
            out['pose_eye'] = pose_eye
        

        if not self.use_vtx_selector:
            # don't need extra joints
            out['Jtr'] = out['Jtr'][:,:self.num_joints+1] # add one for the root

        if not return_dict:
            out = Struct(**out)

        return out

