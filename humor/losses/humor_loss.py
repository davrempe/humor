
import time, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model.body_model import BodyModel
from body_model.utils import SMPLH_PATH, SMPL_JOINTS, VPOSER_PATH, SMPL_PARENTS, KEYPT_VERTS
from utils.transforms import rotation_matrix_to_angle_axis
from datasets.amass_utils import CONTACT_INDS

NUM_BODY_JOINTS = len(SMPL_JOINTS) - 1
BETA_SIZE = 16
CONTACT_THRESH = 0.5

class HumorLoss(nn.Module):

    def __init__(self,
                    kl_loss=1.0,
                    kl_loss_anneal_start=0,
                    kl_loss_anneal_end=0,
                    kl_loss_cycle_len=-1, # if > 0 will anneal KL loss cyclicly
                    regr_trans_loss=1.0,
                    regr_trans_vel_loss=1.0,
                    regr_root_orient_loss=1.0,
                    regr_root_orient_vel_loss=1.0,
                    regr_pose_loss=1.0,
                    regr_pose_vel_loss=1.0,
                    regr_joint_loss=1.0,
                    regr_joint_vel_loss=1.0,
                    regr_joint_orient_vel_loss=1.0,
                    regr_vert_loss=1.0,
                    regr_vert_vel_loss=1.0,
                    contacts_loss=0.0, # classification loss on binary contact prediction
                    contacts_vel_loss=0.0, # velocity near 0 at predicted contacts
                    smpl_joint_loss=0.0,
                    smpl_mesh_loss=0.0,
                    smpl_joint_consistency_loss=0.0,
                    smpl_vert_consistency_loss=0.0,
                    smpl_batch_size=480):
        super(HumorLoss, self).__init__()
        '''
        All loss inputs are weights for that loss term. If the weight is 0, the loss is not used.

        - regr_*_loss :                 L2 regression losses on various state terms (root trans/orient, body pose, joint positions, and joint velocities)
        - smpl_joint_loss :             L2 between GT joints and joint locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_mesh_loss :              L2 between GT and predicted vertex locations resulting from SMPL model (parameterized by trans/orient/body poase)
        - smpl_joint_consistency_loss : L2 between regressed joints and predicted joint locations from SMPL model (ensures consistency between
                                        state joint locations and joint angle predictions)
        - kl_loss :                     divergence between predicted posterior and prior

        - smpl_batch_size : the size of batches that will be given to smpl. if less than this is passed in, will be padded accordingly. however, passed
                            in batches CANNOT be greater than this number.
        '''
        self.kl_loss_weight = kl_loss
        self.kl_loss_anneal_start = kl_loss_anneal_start
        self.kl_loss_anneal_end = kl_loss_anneal_end
        self.use_kl_anneal = self.kl_loss_anneal_end > self.kl_loss_anneal_start

        self.kl_loss_cycle_len = kl_loss_cycle_len
        self.use_kl_cycle = False
        if self.kl_loss_cycle_len > 0:
            self.use_kl_cycle = True
            self.use_kl_anneal = False

        self.contacts_loss_weight = contacts_loss
        self.contacts_vel_loss_weight = contacts_vel_loss
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

        # build dict of all possible regression losses based on inputs
        # keys must be the same as we expect from the pred/gt data
        self.regr_loss_weight_dict = {
            'trans' : regr_trans_loss,
            'trans_vel' : regr_trans_vel_loss,
            'root_orient' : regr_root_orient_loss,
            'root_orient_vel' : regr_root_orient_vel_loss,
            'pose_body' : regr_pose_loss,
            'pose_body_vel' : regr_pose_vel_loss,
            'joints' : regr_joint_loss,
            'joints_vel' : regr_joint_vel_loss,
            'joints_orient_vel' : regr_joint_orient_vel_loss,
            'verts' : regr_vert_loss,
            'verts_vel' : regr_vert_vel_loss
        }

        self.smpl_joint_loss_weight = smpl_joint_loss
        self.smpl_mesh_loss_weight = smpl_mesh_loss
        self.smpl_joint_consistency_loss_weight = smpl_joint_consistency_loss
        self.smpl_vert_consistency_loss_weight = smpl_vert_consistency_loss

        self.l2_loss = nn.MSELoss(reduction='none')
        self.regr_loss = nn.MSELoss(reduction='none')

        smpl_losses = [self.smpl_joint_loss_weight, self.smpl_mesh_loss_weight, self.smpl_joint_consistency_loss_weight, self.smpl_vert_consistency_loss_weight]
        self.smpl_batch_size = smpl_batch_size
        self.use_smpl_losses = False
        if sum(smpl_losses) > 0.0:
            self.use_smpl_losses = True
            # need a body model to compute the losses
            male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
            self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=self.smpl_batch_size)
            female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
            self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=self.smpl_batch_size)

    def forward(self, pred_dict, gt_dict, cur_epoch, gender=None, betas=None):
        '''
        Compute the loss.

        All data in the dictionaries should be of size B x D.

        group_regr_losses will be used to aggregate every group_regr_lossth batch idx together into
        the stats_dict. This can be useful when there are multiple output steps and you want to track
        each separately.
        '''
        loss = 0.0
        stats_dict = dict()

        #
        # KL divergence
        #
        if self.kl_loss_weight > 0.0:
            qm, qv = pred_dict['posterior_distrib']
            pm, pv = pred_dict['prior_distrib']
            kl_loss = self.kl_normal(qm, qv, pm, pv)
            kl_stat_loss = kl_loss.mean()
            kl_loss = kl_stat_loss
            # print(kl_loss.size())
            stats_dict['kl_loss'] = kl_stat_loss
            anneal_weight = 1.0
            if self.use_kl_anneal or self.use_kl_cycle:
                anneal_epoch = cur_epoch
                anneal_start = self.kl_loss_anneal_start
                anneal_end = self.kl_loss_anneal_end
                if self.use_kl_cycle:
                    anneal_epoch = cur_epoch % self.kl_loss_cycle_len
                    anneal_start = 0
                    anneal_end = self.kl_loss_cycle_len // 2 # optimize full weight for second half of cycle
                if anneal_epoch >= anneal_start:
                    anneal_weight = (anneal_epoch - anneal_start) / (anneal_end - anneal_start)
                else:
                    anneal_weight = 0.0
                anneal_weight = 1.0 if anneal_weight > 1.0 else anneal_weight

            loss = loss + anneal_weight*self.kl_loss_weight*kl_loss

            stats_dict['kl_anneal_weight'] = anneal_weight
            stats_dict['kl_weighted_loss'] = loss

        # 
        # Reconstruction 
        #

        # regression terms
        for cur_key in gt_dict.keys():
            # print(cur_key)
            if cur_key not in self.regr_loss_weight_dict:
                continue
            cur_regr_weight = self.regr_loss_weight_dict[cur_key]
            if cur_regr_weight > 0.0:
                pred_val = pred_dict[cur_key]
                gt_val = gt_dict[cur_key]

                if cur_key == 'root_orient' or cur_key == 'pose_body':
                    # rotations use L2 for matrices
                    cur_regr_loss = self.l2_loss(pred_val, gt_val)
                else:
                    cur_regr_loss = self.regr_loss(pred_val, gt_val)
                agg_cur_regr_loss = cur_regr_loss
                cur_regr_stat_loss = agg_cur_regr_loss.mean()
                agg_cur_regr_loss = cur_regr_stat_loss
                stats_dict[cur_key + '_loss'] = cur_regr_stat_loss
                loss = loss + cur_regr_weight*agg_cur_regr_loss

        if self.contacts_loss_weight > 0.0:
            if 'contacts' in gt_dict.keys() and 'contacts' in pred_dict.keys():
                gt_contacts = gt_dict['contacts']
                pred_contacts = pred_dict['contacts']
                # pred is assumed to be logits from network (i.e. sigmoid has not been applied yet)
                cur_contacts_loss = self.bce_loss(pred_contacts, gt_contacts)
                cur_stat_contacts_loss = cur_contacts_loss.mean()
                cur_contacts_loss = cur_stat_contacts_loss
                stats_dict['contacts_loss'] = cur_stat_contacts_loss
                loss = loss + self.contacts_loss_weight*cur_contacts_loss

                # other accuracy statistics
                pred_contacts = (torch.sigmoid(pred_contacts) > CONTACT_THRESH).to(torch.bool)
                gt_contacts = gt_contacts.to(torch.bool)
                # counts for confusion matrix
                # true positive (pred contact, labeled contact)
                true_pos = pred_contacts & gt_contacts
                true_pos_cnt = torch.sum(true_pos).to(torch.float)
                # false positive (pred contact, not lebeled contact)
                false_pos = pred_contacts & ~(gt_contacts)
                false_pos_cnt = torch.sum(false_pos).to(torch.float)
                # false negative (pred no contact, labeled contact)
                false_neg = ~(pred_contacts) & gt_contacts
                false_neg_cnt = torch.sum(false_neg).to(torch.float)
                # true negative (pred no contact, no labeled contact)
                true_neg = (~pred_contacts) & (~gt_contacts)
                true_neg_cnt = torch.sum(true_neg).to(torch.float)

                acc = (true_pos_cnt + true_neg_cnt) / (true_pos_cnt + false_pos_cnt + false_neg_cnt + true_neg_cnt)
                pos_acc = true_pos_cnt / (true_pos_cnt + false_neg_cnt)
                neg_acc = true_neg_cnt / (true_neg_cnt + false_pos_cnt)
                stats_dict['contacts_acc'] = acc
                stats_dict['contacts_pos_acc'] = pos_acc
                stats_dict['contacts_neg_acc'] = neg_acc
            else:
                print('Cannot compute contact loss without contact pred/gt! Skipping...')
            

        if self.contacts_vel_loss_weight > 0.0:
            if 'contacts' in pred_dict.keys() and 'joints_vel' in pred_dict.keys():
                pred_contacts = torch.sigmoid(pred_dict['contacts'])
                pred_joints_vel = pred_dict['joints_vel'].reshape((-1, len(SMPL_JOINTS), 3))
                contact_joints_vel = pred_joints_vel[:,CONTACT_INDS,:]
                # use predicted contact probability to weight regularization on joint velocity
                vel_mag = torch.norm(contact_joints_vel, dim=-1)
                cur_contact_vel_loss = pred_contacts*(vel_mag**2)
                cur_stat_contact_vel_loss = cur_contact_vel_loss.mean()
                cur_contact_vel_loss = cur_stat_contact_vel_loss
                stats_dict['contacts_vel_loss'] = cur_stat_contact_vel_loss
                loss = loss + self.contacts_vel_loss_weight*cur_contact_vel_loss
            else:
                print('Cannot compute contact vel loss without contact and joints_vel pred! Skipping...')

        # terms requiring SMPL reconstruction
        if self.use_smpl_losses:
            if gender is None or betas is None:
                raise Exception('Must pass gender and betas to MotionVAE loss to use SMPL losses!')
            
            try:
                pred_trans = pred_dict['trans']
                pred_orient = pred_dict['root_orient']
                pred_pose = pred_dict['pose_body']
                gt_trans = gt_dict['trans']
                gt_orient = gt_dict['root_orient']
                gt_pose = gt_dict['pose_body']
            except KeyError:
                print('ERROR: In order to use SMPL losses must have trans, root_orient, and pose_body in pred and gt dicts!')
                exit()

            # need to transform rotation matrices to aa for SMPL model
            B = pred_trans.size(0)
            pred_orient = rotation_matrix_to_angle_axis(pred_orient.reshape((B, 3, 3)))
            gt_orient = rotation_matrix_to_angle_axis(gt_orient.reshape((B, 3, 3)))
            pred_pose = rotation_matrix_to_angle_axis(pred_pose.reshape((B*NUM_BODY_JOINTS, 3, 3))).reshape((B, NUM_BODY_JOINTS*3))
            gt_pose = rotation_matrix_to_angle_axis(gt_pose.reshape((B*NUM_BODY_JOINTS, 3, 3))).reshape((B, NUM_BODY_JOINTS*3))

            pred_vals = [pred_trans, pred_orient, pred_pose]
            gt_vals = [gt_trans, gt_orient, gt_pose, betas]

            # have to split by gender to make sure we use the correct body model
            gender_names = ['male', 'female']
            mask_list = []
            pred_joints = []
            pred_mesh = []
            gt_joints = []
            gt_mesh = []
            for gender_name in gender_names:
                # print(gender_name)
                gender_idx = gender[:, 0] == gender_name
                mask_list.append(gender_idx)
                cur_pred_vals = [val[gender_idx] for val in pred_vals] 
                cur_gt_vals = [val[gender_idx] for val in gt_vals]

                # need to pad extra frames with zeros in case not as long as expected 
                pad_size = self.smpl_batch_size - cur_pred_vals[0].size(0)
                if pad_size == self.smpl_batch_size:
                    # skip if no frames for this gender
                    continue
                pad_list = cur_pred_vals + cur_gt_vals
                if pad_size < 0:
                    raise Exception('SMPL model batch size not large enough to accomodate!')
                elif pad_size > 0:
                    pad_list = self.zero_pad_tensors(pad_list, pad_size)
                
                # reconstruct SMPL
                cur_pred_trans, cur_pred_orient, cur_pred_pose, cur_gt_trans, cur_gt_orient, cur_gt_pose, cur_betas = pad_list
                bm = self.male_bm if gender_name == 'male' else self.female_bm
                pred_body = bm(pose_body=cur_pred_pose, betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
                gt_body = bm(pose_body=cur_gt_pose, betas=cur_betas, root_orient=cur_gt_orient, trans=cur_gt_trans)
                if pad_size > 0:
                    pred_joints.append(pred_body.Jtr[:-pad_size])
                    pred_mesh.append(pred_body.v[:-pad_size])
                    gt_joints.append(gt_body.Jtr[:-pad_size])
                    gt_mesh.append(gt_body.v[:-pad_size])
                else:
                    pred_joints.append(pred_body.Jtr)
                    pred_mesh.append(pred_body.v)
                    gt_joints.append(gt_body.Jtr)
                    gt_mesh.append(gt_body.v)

            pred_joints = torch.cat(pred_joints, axis=0)[:,:len(SMPL_JOINTS),:]
            pred_mesh = torch.cat(pred_mesh, axis=0)
            gt_joints = torch.cat(gt_joints, axis=0)[:,:len(SMPL_JOINTS),:]
            gt_mesh = torch.cat(gt_mesh, axis=0)

            pred_verts = pred_mesh[:,KEYPT_VERTS,:]
            gt_verts = gt_mesh[:,KEYPT_VERTS,:]

            # now compute SMPL-related losses
            if self.smpl_joint_loss_weight > 0.0:
                smpl_joint_loss = self.regr_loss(pred_joints, gt_joints)
                smpl_joint_stat_loss = smpl_joint_loss.mean()
                smpl_joint_loss = smpl_joint_stat_loss
                stats_dict['smpl_joint_loss'] = smpl_joint_stat_loss
                loss = loss + self.smpl_joint_loss_weight*smpl_joint_loss
            if self.smpl_mesh_loss_weight > 0.0:
                smpl_mesh_loss = self.regr_loss(pred_mesh, gt_mesh)
                smpl_mesh_stat_loss = smpl_mesh_loss.mean()
                smpl_mesh_loss = smpl_mesh_stat_loss
                stats_dict['smpl_mesh_loss'] = smpl_mesh_stat_loss
                loss = loss + self.smpl_mesh_loss_weight*smpl_mesh_loss
            if self.smpl_joint_consistency_loss_weight > 0.0:
                if not 'joints' in pred_dict.keys():
                    print('Must regress joints in order to use smpl joint consistency loss!')
                    exit()
                regressed_joints = pred_dict['joints'].reshape((B, len(SMPL_JOINTS), -1))
                # need to reorder regressed joints with mask_list to ensure consistency with smpl joints
                regressed_joints = torch.cat([regressed_joints[mask_list[i]] for i in range(len(mask_list))], axis=0)

                smpl_joint_consistency_loss = self.regr_loss(pred_joints, regressed_joints)
                smpl_joint_consistency_stat_loss = smpl_joint_consistency_loss.mean()
                smpl_joint_consistency_loss = smpl_joint_consistency_stat_loss
                stats_dict['smpl_joint_consistency_loss'] = smpl_joint_consistency_stat_loss
                loss = loss + self.smpl_joint_consistency_loss_weight*smpl_joint_consistency_loss
            if self.smpl_vert_consistency_loss_weight > 0.0:
                if not 'verts' in pred_dict.keys():
                    print('Must regress verts in order to use smpl vert consistency loss!')
                    exit()
                regressed_verts = pred_dict['verts'].reshape((B, len(KEYPT_VERTS), -1))
                # need to reorder regressed verts with mask_list to ensure consistency with smpl verts
                regressed_verts = torch.cat([regressed_verts[mask_list[i]] for i in range(len(mask_list))], axis=0)

                smpl_vert_consistency_loss = self.regr_loss(pred_verts, regressed_verts)
                smpl_vert_consistency_stat_loss = smpl_vert_consistency_loss.mean()
                smpl_vert_consistency_loss = smpl_vert_consistency_stat_loss
                stats_dict['smpl_vert_consistency_loss'] = smpl_vert_consistency_stat_loss
                loss = loss + self.smpl_vert_consistency_loss_weight*smpl_vert_consistency_loss

        if self.kl_loss_weight > 0.0:
            stats_dict['reconstr_weighted_loss'] = loss - stats_dict['kl_weighted_loss']
        
        return loss, stats_dict

    def zero_pad_tensors(self, pad_list, pad_size):
        '''
        Assumes tensors in pad_list are B x D
        '''
        new_pad_list = []
        for pad_idx, pad_tensor in enumerate(pad_list):
            padding = torch.zeros((pad_size, pad_tensor.size(1))).to(pad_tensor)
            new_pad_list.append(torch.cat([pad_tensor, padding], dim=0))
        return new_pad_list

    
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

    