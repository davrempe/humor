import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logging import Logger
from torch.distributions import MixtureSameFamily, Categorical, Normal, MultivariateNormal

from fitting.fitting_utils import OP_NUM_JOINTS, perspective_projection, apply_robust_weighting, gmof

from body_model.utils import SMPL_JOINTS, SMPL_PARENTS

CONTACT_HEIGHT_THRESH = 0.08

class FittingLoss(nn.Module):
    '''
    Functions to compute all needed losses for fitting.
    '''

    def __init__(self, loss_weights,
                       init_motion_prior=None,
                       smpl2op_map=None,
                       ignore_op_joints=None,
                       cam_f=None, 
                       cam_cent=None, 
                       robust_loss='none',
                       robust_tuning_const=4.6851,
                       joints2d_sigma=100,
                       use_chamfer=False):
        super(FittingLoss, self).__init__()
        self.all_stage_loss_weights = loss_weights
        self.cur_stage_idx = 0
        self.loss_weights = self.all_stage_loss_weights[self.cur_stage_idx]
        self.smpl2op_map = smpl2op_map
        self.ignore_op_joints = ignore_op_joints
        self.cam_f = cam_f
        self.cam_cent = cam_cent
        self.joints2d_sigma = joints2d_sigma

        self.can_reproj = self.smpl2op_map is not None and \
                          self.cam_f is not None and \
                          self.cam_cent is not None
        if self.can_reproj:
            self.cam_f = self.cam_f.reshape((-1, 1, 2))
            self.cam_cent = self.cam_cent.reshape((-1, 1, 2))

        if use_chamfer:
            from utils.chamfer_distance import ChamferDistance
            self.chamfer_dist = ChamferDistance()

        sum_loss_weights = {k : 0.0 for k in self.loss_weights.keys()}
        for stage_idx in range(len(self.all_stage_loss_weights)):
            for k in sum_loss_weights.keys():
                sum_loss_weights[k] += self.all_stage_loss_weights[stage_idx][k]

        self.init_motion_prior = None
        if init_motion_prior is not None and sum_loss_weights['init_motion_prior'] > 0.0:
            self.init_motion_prior = dict()
            # build pytorch GMM 
            gmm = self.build_init_motion_prior_distrib(*init_motion_prior['gmm'])
            self.init_motion_prior['gmm'] = gmm

        self.l2_loss = nn.MSELoss(reduction='none')
        self.robust_loss = robust_loss
        self.robust_tuning_const = robust_tuning_const
        robust_choices = ['none', 'bisquare', 'gm']
        if self.robust_loss not in robust_choices:
            Logger.log('Not a valid robust loss: %s. Please use %s' % (self.robust_loss, str(robust_choices)))
            exit()

        self.cur_optim_step = 0

    def set_stage(self, idx):
        ''' Sets the current stage index. Determines which loss weights are used '''
        self.cur_stage_idx = idx
        self.loss_weights = self.all_stage_loss_weights[self.cur_stage_idx]
        Logger.log('Stage %d loss weights set to:' % (idx))
        Logger.log(self.loss_weights)

    def build_init_motion_prior_distrib(self, gmm_weights, gmm_means, gmm_covs):
        mix = Categorical(gmm_weights)
        comp = MultivariateNormal(gmm_means, covariance_matrix=gmm_covs)
        gmm_distrib = MixtureSameFamily(mix, comp)
        return gmm_distrib

    def forward(self):
        pass

    def root_fit(self, observed_data, pred_data):
        '''
        For fitting just global root trans/orientation. Only computes joint/point/vert losses, i.e. no priors.
        '''
        stats_dict = dict()
        loss = 0.0

        # Joints in 3D space
        if 'joints3d' in observed_data and 'joints3d' in pred_data and self.loss_weights['joints3d'] > 0.0:
            cur_loss = self.joints3d_loss(observed_data['joints3d'], pred_data['joints3d'])
            loss += self.loss_weights['joints3d']*cur_loss
            stats_dict['joints3d'] = cur_loss

        # Select vertices in 3D space
        if 'verts3d' in observed_data and 'verts3d' in pred_data and self.loss_weights['verts3d'] > 0.0:
            cur_loss = self.verts3d_loss(observed_data['verts3d'], pred_data['verts3d'])
            loss += self.loss_weights['verts3d']*cur_loss
            stats_dict['verts3d'] = cur_loss

        # All vertices to non-corresponding observed points in 3D space
        if 'points3d' in observed_data and 'points3d' in pred_data and self.loss_weights['points3d'] > 0.0:
            cur_loss = self.points3d_loss(observed_data['points3d'], pred_data['points3d'])
            loss += self.loss_weights['points3d']*cur_loss
            stats_dict['points3d'] = cur_loss

        # 2D re-projection loss
        use_reproj_loss = 'joints2d' in observed_data and \
                          'joints3d' in pred_data and \
                          'joints3d_extra' in pred_data and \
                          self.loss_weights['joints2d'] > 0.0
        if use_reproj_loss:
            if not self.can_reproj:
                Logger.log('Must provide camera intrinsics and SMPL to OpenPose joint map to use re-projection loss!')
                exit()
            cur_loss = self.joints2d_loss(observed_data['joints2d'],
                                          pred_data['joints3d'],
                                          pred_data['joints3d_extra'],
                                          debug_img=None)
            loss += self.loss_weights['joints2d']*cur_loss
            stats_dict['joints2d'] = cur_loss

        # verts3d consistency loss across batch indices
        if 'seq_interval' in observed_data and 'verts3d' in pred_data and self.loss_weights['rgb_overlap_consist'] > 0.0:
            # NOTE this is only correct when FULL sequences are being used, i.e. the seq_intervals actually correctly
            #       correspond to the given pred and obs data
            overlap_lens = observed_data['seq_interval'][:-1, 1] - observed_data['seq_interval'][1:, 0]
            cur_pos_loss = 0.0
            cur_vel_loss = 0.0
            for bidx in range(1, pred_data['verts3d'].size(0)):
                cur_ov = overlap_lens[bidx-1]
                # position
                prev_pos = pred_data['verts3d'][bidx-1:bidx, -cur_ov:]
                cur_pos = pred_data['verts3d'][bidx:bidx+1, :cur_ov]
                cur_pos_loss += self.verts3d_loss(prev_pos, cur_pos)
                if cur_ov > 1:
                    # velocity
                    prev_vel = prev_pos[:, 1:] - prev_pos[:, :-1]
                    cur_vel = cur_pos[:, 1:] - cur_pos[:, :-1]
                    cur_vel_loss += self.verts3d_loss(prev_vel, cur_vel)

            loss += self.loss_weights['rgb_overlap_consist']*cur_pos_loss
            stats_dict['rgb_overlap_consist_verts3d_pos'] = cur_pos_loss
            loss += self.loss_weights['rgb_overlap_consist']*cur_vel_loss
            stats_dict['rgb_overlap_consist_verts3d_vel'] = cur_vel_loss

            if 'prev_batch_overlap_res' in observed_data:
                # make sure first batch is consistent with previous last batch
                cur_ov = (observed_data['prev_batch_overlap_res']['seq_interval'][1] - observed_data['seq_interval'][0, 0]).cpu().item()
                # our current length may be less than the overlap (if using e.g. first 15 frames), account for this
                cur_pred_len = pred_data['verts3d'].size(1)
                ov_len = cur_pred_len if cur_pred_len < cur_ov else cur_ov
                # positions
                prev_pos = observed_data['prev_batch_overlap_res']['verts3d'][-cur_ov:][:ov_len]
                cur_pos = pred_data['verts3d'][0, :ov_len]
                cur_pos_loss = self.verts3d_loss(prev_pos, cur_pos)
                # velocity
                cur_vel_loss = 0.0
                if cur_ov > 1:
                    prev_vel = prev_pos[1:] - prev_pos[:-1]
                    cur_vel = cur_pos[1:] - cur_pos[:-1]
                    cur_vel_loss = self.verts3d_loss(prev_vel, cur_vel)
                
                loss += self.loss_weights['rgb_overlap_consist']*cur_pos_loss
                stats_dict['rgb_overlap_xbatch_verts3d_pos'] = cur_pos_loss
                loss += self.loss_weights['rgb_overlap_consist']*cur_vel_loss
                stats_dict['rgb_overlap_xbatch_verts3d_vel'] = cur_vel_loss

        return loss, stats_dict
        
    def smpl_fit(self, observed_data, pred_data, nsteps):
        '''
        For fitting full shape and pose of SMPL.

        nsteps used to scale single-step losses
        '''
        # first all observation losses
        loss, stats_dict = self.root_fit(observed_data, pred_data)

        # prior to keep latent pose likely
        if 'latent_pose' in pred_data and self.loss_weights['pose_prior'] > 0.0:
            cur_loss = self.pose_prior_loss(pred_data['latent_pose'])
            loss += self.loss_weights['pose_prior']*cur_loss
            stats_dict['pose_prior'] = cur_loss

        # prior to keep PCA shape likely
        if 'betas' in pred_data and self.loss_weights['shape_prior'] > 0.0:
            cur_loss = self.shape_prior_loss(pred_data['betas'])
            loss += self.loss_weights['shape_prior']*nsteps*cur_loss
            stats_dict['shape_prior'] = cur_loss

        # smooth 3d joint motion
        if self.loss_weights['joints3d_smooth'] > 0.0:
            cur_loss = self.joints3d_smooth_loss(pred_data['joints3d'])
            loss += self.loss_weights['joints3d_smooth']*cur_loss
            stats_dict['joints3d_smooth'] = cur_loss

        # shape consistency loss across batch
        if 'seq_interval' in observed_data and 'betas' in pred_data and self.loss_weights['rgb_overlap_consist'] > 0.0:
            cur_loss = self.joints3d_loss(pred_data['betas'][:-1, :],
                                            pred_data['betas'][1:, :])
            loss += self.loss_weights['rgb_overlap_consist']*cur_loss
            stats_dict['rgb_overlap_consist_betas'] = cur_loss

            if 'prev_batch_overlap_res' in observed_data:
                # make sure first batch is consistent with previous last batch
                cur_loss = self.joints3d_loss(pred_data['betas'][0, :],
                                            observed_data['prev_batch_overlap_res']['betas'])
                loss += self.loss_weights['rgb_overlap_consist']*cur_loss
                stats_dict['rgb_overlap_xbatch_betas'] = cur_loss

        return loss, stats_dict

    def motion_fit(self, observed_data, pred_data, cam_pred_data, nsteps, cond_prior=None, init_motion_scale=1.0):
        '''
        For fitting full shape and pose of SMPL with motion prior.

        pred_data is data pertinent to the canonical prior coordinate frame
        cam_pred_data is for the camera coordinate frame

        cond_prior is a tuple of (mean, variance) that is used to compute the prior
        loss rather than standard normal if given.
        '''
        # first shape/pose and observations
        loss, stats_dict = self.smpl_fit(observed_data, cam_pred_data, nsteps)

        # prior to keep latent motion likely
        if 'latent_motion' in pred_data and self.loss_weights['motion_prior'] > 0.0:
            cur_loss = self.motion_prior_loss(pred_data['latent_motion'], cond_prior=cond_prior)
            loss += self.loss_weights['motion_prior']*cur_loss
            stats_dict['motion_prior'] = cur_loss

        # prior to keep initial state likely
        have_init_prior_info = 'joints3d' in pred_data and \
                               'joints_vel' in pred_data and \
                               'trans_vel' in pred_data and \
                               'root_orient_vel' in pred_data
        if have_init_prior_info and self.loss_weights['init_motion_prior'] > 0.0:
            cur_loss = self.init_motion_prior_loss(pred_data['joints3d'][:,0:1], # only need first step
                                                   pred_data['joints_vel'],
                                                   pred_data['trans_vel'],
                                                   pred_data['root_orient_vel'])
            loss += self.loss_weights['init_motion_prior']*init_motion_scale*cur_loss # must scale since doesn't scale with more steps
            stats_dict['init_motion_prior'] = cur_loss

        # make sure joints consistent between SMPL and direct motion prior output
        if 'joints3d_rollout' in pred_data and 'joints3d' in pred_data and self.loss_weights['joint_consistency'] > 0.0:
            cur_loss = self.joint_consistency_loss(pred_data['joints3d'], pred_data['joints3d_rollout'])
            loss += self.loss_weights['joint_consistency']*cur_loss
            stats_dict['joint_consistency'] = cur_loss

        # make sure bone lengths between frames of direct motion prior output are consistent
        if 'joints3d_rollout' in pred_data and self.loss_weights['bone_length'] > 0.0:
            cur_loss = self.bone_length_loss(pred_data['joints3d_rollout'])
            loss += self.loss_weights['bone_length']*cur_loss
            stats_dict['bone_length'] = cur_loss

        # make sure rolled out joints match observations too
        if 'joints3d' in observed_data and 'joints3d_rollout' in pred_data and self.loss_weights['joints3d_rollout'] > 0.0:
            cur_loss = self.joints3d_loss(observed_data['joints3d'], pred_data['joints3d_rollout'])
            loss += self.loss_weights['joints3d_rollout']*cur_loss
            stats_dict['joints3d_rollout'] = cur_loss

        # velocity 0 during contacts
        if self.loss_weights['contact_vel'] > 0.0 and 'contacts_conf' in pred_data and 'joints3d' in pred_data:
            cur_loss = self.contact_vel_loss(pred_data['contacts_conf'], pred_data['joints3d'])
            loss += self.loss_weights['contact_vel']*cur_loss
            stats_dict['contact_vel'] = cur_loss

        # contacting joints are near the floor
        if self.loss_weights['contact_height'] > 0.0 and 'contacts_conf' in pred_data and 'joints3d' in pred_data:
            cur_loss = self.contact_height_loss(pred_data['contacts_conf'], pred_data['joints3d'])
            loss += self.loss_weights['contact_height']*cur_loss
            stats_dict['contact_height'] = cur_loss

        # floor is close to the initialization
        if self.loss_weights['floor_reg'] > 0.0 and 'floor_plane' in cam_pred_data and 'floor_plane' in observed_data:
            cur_loss = self.floor_reg_loss(cam_pred_data['floor_plane'], observed_data['floor_plane'])
            loss += self.loss_weights['floor_reg']*nsteps*cur_loss
            stats_dict['floor_reg'] = cur_loss
            pass

        # floor consistency loss across batch
        if 'seq_interval' in observed_data and 'floor_plane' in cam_pred_data and self.loss_weights['rgb_overlap_consist'] > 0.0:
            cur_loss = self.joints3d_loss(cam_pred_data['floor_plane'][:-1, :],
                                            cam_pred_data['floor_plane'][1:, :])
            loss += self.loss_weights['rgb_overlap_consist']*cur_loss
            stats_dict['rgb_overlap_consist_floor'] = cur_loss

            if 'prev_batch_overlap_res' in observed_data:
                # make sure first batch is consistent with previous last batch
                cur_loss = self.floor_reg_loss(cam_pred_data['floor_plane'][0:1, :],
                                            observed_data['prev_batch_overlap_res']['floor_plane'].unsqueeze(0))
                loss += self.loss_weights['rgb_overlap_consist']*cur_loss
                stats_dict['rgb_overlap_xbatch_floor'] = cur_loss

        return loss, stats_dict

    def get_visible_mask(self, obs_data):
        '''
        Given observed data gets the mask of visible data (that actually contributes to the loss).
        '''
        return torch.logical_not(torch.isinf(obs_data))

    def joints2d_loss(self, joints2d_obs, joints3d_pred, joints3d_extra_pred, cam_t=None, cam_R=None, debug_img=None):
        '''
        Cam extrinsics are assumed the same for entire sequence
        - cam_t : (B, 3)
        - cam_R : (B, 3, 3)
        '''
        B, T, _, _ = joints2d_obs.size()
        # need extra joints that correspond to openpose
        joints3d_full = torch.cat([joints3d_pred, joints3d_extra_pred], dim=2)
        joints3d_op = joints3d_full[:,:,self.smpl2op_map,:]
        joints3d_op = joints3d_op.reshape((B*T, OP_NUM_JOINTS, 3))

        # either use identity cam params or expand the ones given to full sequence
        if cam_t is None:
            cam_t = torch.zeros((B*T, 3)).to(joints3d_pred)
        else:
            cam_t = cam_t.unsqueeze(1).expand((B, T, 3)).reshape((B*T, 3))
        if cam_R is None:
            cam_R = torch.eye(3).reshape((1, 3, 3)).expand((B*T, 3, 3)).to(joints3d_pred)
        else:
            cam_R = cam_R.unsqueeze(1).expand((B, T, 3, 3)).reshape((B*T, 3, 3))

        # project points to 2D
        cam_f = self.cam_f.expand((B, T, 2)).reshape((B*T, 2))
        cam_cent = self.cam_cent.expand((B, T, 2)).reshape((B*T, 2))
        joints2d_pred = perspective_projection(joints3d_op,
                                               cam_R,
                                               cam_t,
                                               cam_f,
                                               cam_cent)

        # compared to observations
        joints2d_pred = joints2d_pred.reshape((B, T, OP_NUM_JOINTS, 2))
        joints2d_obs_conf = joints2d_obs[:,:,:,2:3]
        if self.ignore_op_joints is not None:
            joints2d_obs_conf[:,:,self.ignore_op_joints] = 0.0 # set confidence to 0 so not weighted

        # weight errors by detection confidence
        robust_sqr_dist = gmof(joints2d_pred - joints2d_obs[:,:,:,:2], self.joints2d_sigma)
        reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
        loss = torch.sum(reproj_err)
        return loss

    def joints3d_loss(self, joints3d_obs, joints3d_pred):
        vis_mask = self.get_visible_mask(joints3d_obs)
        loss = (joints3d_obs[vis_mask] - joints3d_pred[vis_mask])**2
        loss = 0.5*torch.sum(loss)
        return loss

    def joints3d_smooth_loss(self, joints3d_pred):
        # minimize delta steps
        loss = (joints3d_pred[:,1:,:,:] - joints3d_pred[:,:-1,:,:])**2
        loss = 0.5*torch.sum(loss)
        return loss

    def verts3d_loss(self, verts3d_obs, verts3d_pred):
        vis_mask = self.get_visible_mask(verts3d_obs)
        loss = (verts3d_obs[vis_mask] - verts3d_pred[vis_mask])**2
        loss = 0.5*torch.sum(loss)
        return loss

    def points3d_loss(self, points3d_obs, points3d_pred):
        # one-way chamfer
        B, T, N_obs, _ = points3d_obs.size()
        N_pred = points3d_pred.size(2)
        points3d_obs = points3d_obs.reshape((B*T, -1, 3))
        points3d_pred = points3d_pred.reshape((B*T, -1, 3))

        obs2pred_sqr_dist, pred2obs_sqr_dist = self.chamfer_dist(points3d_obs, points3d_pred)
        obs2pred_sqr_dist = obs2pred_sqr_dist.reshape((B, T*N_obs))
        pred2obs_sqr_dist = pred2obs_sqr_dist.reshape((B, T*N_pred))


        weighted_obs2pred_sqr_dist, w = apply_robust_weighting(obs2pred_sqr_dist.sqrt(),
                                                        robust_loss_type=self.robust_loss,
                                                        robust_tuning_const=self.robust_tuning_const)

        loss = torch.sum(weighted_obs2pred_sqr_dist)
        loss = 0.5*loss
        return loss

    def pose_prior_loss(self, latent_pose_pred):
        # prior is isotropic gaussian so take L2 distance from 0
        loss = latent_pose_pred**2
        loss = torch.sum(loss)
        return loss

    def motion_prior_loss(self, latent_motion_pred, cond_prior=None):
        if cond_prior is None:
            # assume standard normal
            loss = latent_motion_pred**2
            loss = torch.sum(loss)
        else:
            pm, pv = cond_prior
            loss = -self.log_normal(latent_motion_pred, pm, pv)
            loss = torch.sum(loss)

        return loss

    def init_motion_prior_loss(self, joints, joints_vel, trans_vel, root_orient_vel):
        # create input
        B = joints.size(0)

        joints = joints.reshape((B, -1))
        joints_vel = joints_vel.reshape((B, -1))
        trans_vel = trans_vel.reshape((B, -1))
        root_orient_vel = root_orient_vel.reshape((B, -1))
        init_state = torch.cat([joints, joints_vel, trans_vel, root_orient_vel], dim=-1)

        loss = -self.init_motion_prior['gmm'].log_prob(init_state)
        loss = torch.sum(loss)

        return loss

    def joint_consistency_loss(self, smpl_joints3d, rollout_joints3d):
        loss = (smpl_joints3d - rollout_joints3d)**2
        loss = 0.5*torch.sum(loss)
        return loss

    def bone_length_loss(self, rollout_joints3d):
        bones = rollout_joints3d[:,:,1:]
        parents = rollout_joints3d[:,:,SMPL_PARENTS[1:]]
        bone_lengths = torch.norm(bones - parents, dim=-1)
        bone_length_diff = bone_lengths[:,1:] - bone_lengths[:,:-1]
        loss = 0.5*torch.sum(bone_length_diff**2)
        return loss

    def shape_prior_loss(self, betas_pred):
        # prior is isotropic gaussian so take L2 distance from 0
        loss = betas_pred**2
        loss = torch.sum(loss)
        return loss

    def contact_vel_loss(self, contacts_conf, joints3d):
        '''
        Velocity should be zero at predicted contacts
        '''
        delta_pos = (joints3d[:,1:] - joints3d[:,:-1])**2
        cur_loss = delta_pos.sum(dim=-1) * contacts_conf[:,1:]
        cur_loss = 0.5*torch.sum(cur_loss)

        return cur_loss

    def contact_height_loss(self, contacts_conf, joints3d):
        '''
        Contacting joints should be near floor
        '''
        # won't be exactly on the floor, just near it (since joints are inside the body)
        floor_diff = F.relu(torch.torch.abs(joints3d[:,:,:,2]) - CONTACT_HEIGHT_THRESH)
        cur_loss = floor_diff * contacts_conf
        cur_loss = torch.sum(cur_loss)

        return cur_loss

    def floor_reg_loss(sefl, pred_floor_plane, obs_floor_plane):
        '''
        Pred floor plane shouldn't deviate from the initial observation
        '''
        # pred floor plane is 3d param, observed floor plane is 4d param
        # convert obs to 3d
        obs_normal = obs_floor_plane[:,:3]
        obs_offset = obs_floor_plane[:,3:]
        obs_floor_plane = obs_normal * obs_offset
        # then compare
        floor_loss = (pred_floor_plane - obs_floor_plane)**2
        floor_loss = 0.5*torch.sum(floor_loss)

        return floor_loss

    def kl_normal(self, qm, qv, pm, pv):
        """
        Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
        sum over the last dimension
        ​
        Args:
            qm: tensor: (batch, dim): q mean
            qv: tensor: (batch, dim): q variance
            pm: tensor: (batch, dim): p mean
            pv: tensor: (batch, dim): p variance
        ​
        Return:
            kl: tensor: (batch,): kl between each sample
        """
        element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
        kl = element_wise.sum(-1)
        return kl

    def log_normal(self, x, m, v):
        """
        Computes the elem-wise log probability of a Gaussian and then sum over the
        last dim. Basically we're assuming all dims are batch dims except for the
        last dim.    Args:
            x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
            m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
            v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance    Return:
            log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
                each sample. Note that the summation dimension is not kept
        """
        log_prob = -torch.log(torch.sqrt(v)) - math.log(math.sqrt(2*math.pi)) \
                        - ((x - m)**2 / (2*v))
        log_prob = torch.sum(log_prob, dim=-1)
        return log_prob