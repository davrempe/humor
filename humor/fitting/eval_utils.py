import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import os
import numpy as np
import torch

from datasets.amass_utils import CONTACT_INDS
from body_model.utils import SMPL_JOINTS
from fitting.fitting_utils import perspective_projection, bdot, compute_plane_intersection
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues

SMPL_SIZES = {
    'trans' : 3,
    'betas' : 10,
    'pose_body' : 63,
    'root_orient' : 3
}

GRND_PEN_THRESH_LIST = [0.0, 0.03, 0.06, 0.09, 0.12, 0.15]
IMW, IMH = 1920, 1080 # all data
DATA_FPS = 30.0
DATA_h = 1.0 / DATA_FPS

# Baseline MVAE does not converge on these sequences, don't use for quantitative evaluation
AMASS_EVAL_BLACKLIST = [ 
    'HumanEva_S1_Box_1_poses_548_frames_30_fps',
    'HumanEva_S1_Box_3_poses_330_frames_30_fps',
    'HumanEva_S1_Gestures_1_poses_594_frames_30_fps'
]

# PROX-D baseline catastrophically fails on these, don't use for quantitative evaluation
RGBD_EVAL_BLACKLIST = [
    'MPH1Library_00145_01_0020',
    'MPH1Library_00145_01_0021',
    'MPH1Library_00145_01_0022',
    'MPH1Library_00145_01_0023',
    'MPH1Library_00145_01_0024',
    'MPH1Library_00145_01_0025',
    'MPH1Library_00145_01_0026',
    'MPH1Library_00145_01_0027',
    'MPH1Library_00145_01_0028',
    'N0Sofa_03403_01_0000',
    'N0Sofa_03403_01_0001',
    'N0Sofa_03403_01_0002',
    'N0Sofa_03403_01_0003',
    'N0Sofa_03403_01_0004',
    'N0Sofa_03403_01_0005',
    'N0Sofa_03403_01_0006',
    'N0Sofa_03403_01_0007',
    'N0Sofa_03403_01_0008',
    'N0Sofa_03403_01_0009',
    'N0Sofa_03403_01_0010',
    'N0Sofa_03403_01_0011',
    'N0Sofa_03403_01_0012',
    'N0Sofa_03403_01_0013',
    'N0Sofa_03403_01_0014'
]

# VIBE baseline catastrophically fails on these, don't use for quantitative evaluation
RGB_EVAL_BLACKLIST = [
    'MPH1Library_00145_01_0031',
    'N0Sofa_03403_01_0013'
]


def get_grnd_pen_key(thresh):
    return 'ground_pen@%0.2f' % (thresh)

def quant_eval_3d(eval_dict, pred_data, gt_data, obs_data):
    # get positional errors for each modality
    for modality in ['joints3d', 'verts3d', 'mesh3d']:
        eval_pred = pred_data[modality]
        eval_gt = gt_data[modality]

        # all positional errors
        pos_err_all = torch.norm(eval_pred - eval_gt, dim=-1).detach().cpu().numpy()
        eval_dict[modality + '_all'].append(pos_err_all)

        # ee and legs
        if modality == 'joints3d':
            joints3d_ee = compute_subset_smpl_joint_err(eval_pred, eval_gt, subset='ee').detach().cpu().numpy()
            joints3d_legs = compute_subset_smpl_joint_err(eval_pred, eval_gt, subset='legs').detach().cpu().numpy()
            eval_dict['joints3d_ee'].append(joints3d_ee)
            eval_dict['joints3d_legs'].append(joints3d_legs)

        # split by occluded/visible if this was the observed modality
        if modality in obs_data:
            # visible data (based on observed)
            eval_obs = obs_data[modality]
            invis_mask = torch.isinf(eval_obs)
            vis_mask = torch.logical_not(invis_mask) # T x N x 3
            num_invis_pts = torch.sum(invis_mask[:,:,0])
            num_vis_pts = torch.sum(vis_mask[:,:,0])

            if num_vis_pts > 0:
                pred_vis = eval_pred[vis_mask].reshape((num_vis_pts, 3))
                gt_vis = eval_gt[vis_mask].reshape((num_vis_pts, 3))
                vis_err = torch.norm(pred_vis - gt_vis, dim=-1).detach().cpu().numpy()
                eval_dict[modality + '_vis'].append(vis_err)
            else:
                eval_dict[modality + '_vis'].append(np.zeros((0)))

            # invisible data 
            if num_invis_pts > 0:
                pred_invis = eval_pred[invis_mask].reshape((num_invis_pts, 3))
                gt_invis = eval_gt[invis_mask].reshape((num_invis_pts, 3))
                invis_err = torch.norm(pred_invis - gt_invis, dim=-1).detach().cpu().numpy()
                eval_dict[modality + '_occ'].append(invis_err)
            else:
                eval_dict[modality + '_occ'].append(np.zeros((0)))

    # per-joint acceleration mag
    pred_joint_accel, pred_joint_accel_mag = compute_joint_accel(pred_data['joints3d'])
    eval_dict['accel_mag'].append(pred_joint_accel_mag.detach().cpu().numpy())

    # toe-floor penetration
    floor_plane = torch.zeros((4)).to(pred_data['joints3d'])
    floor_plane[2] = 1.0
    num_pen_list, num_tot, pen_dist = compute_toe_floor_pen(pred_data['joints3d'], floor_plane, thresh_list=GRND_PEN_THRESH_LIST)
    eval_dict['ground_pen_dist'].append(pen_dist.detach().cpu().numpy())
    for thresh_idx, pen_thresh in enumerate(GRND_PEN_THRESH_LIST):
        cur_pen_key = get_grnd_pen_key(pen_thresh)
        eval_dict[cur_pen_key].append(num_pen_list[thresh_idx].detach().cpu().item())
        eval_dict[cur_pen_key + '_cnt'].append(num_tot)

    # contact classification (output number correct and total frame cnt)
    pred_contacts = pred_data['contacts'][:,CONTACT_INDS] # only compare joints for which the prior is trained
    gt_contacts = gt_data['contacts'][:,CONTACT_INDS]
    num_correct = np.sum((pred_contacts - gt_contacts) == 0)
    total_cnt = pred_contacts.shape[0]*pred_contacts.shape[1]
    eval_dict['contact_acc'].append(num_correct)
    eval_dict['contact_acc_cnt'].append(total_cnt)  


def quant_eval_2d(eval_dict, pred_joints_smpl, floor_plane, 
                  pred_joints_comp=None,
                  gt_joints_comp=None,
                  vis_mask=None,
                  cam_intrins=None):
    '''
    eval_dict dictionary accumulator to add results to.

    Always must have pred_joints_smpl and floor_plane to compute plausibility metrics.

    Optionally comparison pred and gt joints along with a joint occlusion mask can be passed in
    to compute joint position errors. Errors will also be split by visible/occluded.

    all_joints T x J x 3
    floor_plane (4)
    '''

    do_comparison = pred_joints_comp is not None and \
                    gt_joints_comp is not None

    if do_comparison:
        #
        # pointwise distance errors (MPJPE)
        # split by visiblity if observations are available
        #
        T, J, _ = gt_joints_comp.size()
        entires_per_frame = J*3

        # mask out missing frames
        invalid_mask = torch.isinf(gt_joints_comp)
        num_inval_entries = torch.sum(invalid_mask, dim=[1, 2])
        valid_mask = (num_inval_entries < entires_per_frame)
        num_val_frames = torch.sum(valid_mask)
        valid_frame_mask = valid_mask.reshape((T, 1, 1)).expand_as(gt_joints_comp)

        eval_pred_joints = pred_joints_comp[valid_frame_mask].reshape((num_val_frames, J, 3))
        eval_gt_joints = gt_joints_comp[valid_frame_mask].reshape((num_val_frames, J, 3))

        # all joint errors
        joints3d_all = torch.norm(eval_pred_joints - eval_gt_joints, dim=-1).detach().cpu().numpy()
        eval_dict['joints3d_all'].append(joints3d_all)

        # end-effector and related errors
        joints3d_ee = compute_subset_comp_joint_err(eval_pred_joints, eval_gt_joints, subset='ee').detach().cpu().numpy()
        joints3d_legs = compute_subset_comp_joint_err(eval_pred_joints, eval_gt_joints, subset='legs').detach().cpu().numpy()
        eval_dict['joints3d_ee'].append(joints3d_ee)
        eval_dict['joints3d_legs'].append(joints3d_legs)

        # aligned at root all joint errors
        pred_root = eval_pred_joints[:, COMP_ROOT_IDX:(COMP_ROOT_IDX+1), :]
        eval_pred_joints_align = eval_pred_joints - pred_root
        gt_root = eval_gt_joints[:, COMP_ROOT_IDX:(COMP_ROOT_IDX+1), :]
        eval_gt_joints_align = eval_gt_joints - gt_root
        joints3d_align_all = torch.norm(eval_pred_joints_align - eval_gt_joints_align, dim=-1).detach().cpu().numpy()
        eval_dict['joints3d_align_all'].append(joints3d_align_all)

        # end-effector and related errors
        joints3d_align_ee = compute_subset_comp_joint_err(eval_pred_joints_align, eval_gt_joints_align, subset='ee').detach().cpu().numpy()
        joints3d_align_legs = compute_subset_comp_joint_err(eval_pred_joints_align, eval_gt_joints_align, subset='legs').detach().cpu().numpy()
        eval_dict['joints3d_align_ee'].append(joints3d_align_ee)
        eval_dict['joints3d_align_legs'].append(joints3d_align_legs)
        
        if vis_mask is not None and cam_intrins is not None:
            # split into visible and occluded
            valid_vis_masks = vis_mask[valid_mask.cpu().numpy()]

            Tv = num_val_frames
            cam_t = torch.zeros((Tv, 3)).to(eval_gt_joints)
            cam_R = torch.eye(3).reshape((1, 3, 3)).expand((Tv, 3, 3)).to(eval_gt_joints)
            # project points to 2D
            cam_f = torch.zeros((Tv, 2)).to(eval_gt_joints)
            cam_f[:,0] = float(cam_intrins[0])
            cam_f[:,1] = float(cam_intrins[1])
            cam_cent = torch.zeros((Tv, 2)).to(eval_gt_joints)
            cam_cent[:,0] = float(cam_intrins[2])
            cam_cent[:,1] = float(cam_intrins[3])
            gt_joints2d = perspective_projection(eval_gt_joints,
                                                cam_R,
                                                cam_t,
                                                cam_f,
                                                cam_cent)

            uvs = np.round(gt_joints2d.cpu().numpy()).astype(int)
            uvs[:,:,0] = np.clip(uvs[:,:,0], 0, IMW-1)
            uvs[:,:,1] = np.clip(uvs[:,:,1], 0, IMH-1)
            occlusion_mask = np.zeros((Tv, J), dtype=np.bool)
            for t in range(Tv):
                occlusion_mask[t] = valid_vis_masks[t][uvs[t, :, 1], uvs[t, :, 0]] == 1

            occlusion_mask = torch.Tensor(occlusion_mask).to(torch.bool).to(eval_gt_joints.device)
            vis_mask = torch.logical_not(occlusion_mask)
            num_occl_pts = torch.sum(occlusion_mask)
            num_vis_pts = torch.sum(vis_mask)

            occlusion_mask = occlusion_mask.unsqueeze(2).expand_as(eval_gt_joints)
            vis_mask = vis_mask.unsqueeze(2).expand_as(eval_gt_joints)

            # visible absolute data
            if num_vis_pts > 0:
                eval_pred_vis = eval_pred_joints[vis_mask].reshape((num_vis_pts, 3))
                eval_gt_vis = eval_gt_joints[vis_mask].reshape((num_vis_pts, 3))
                joints3d_vis = torch.norm(eval_pred_vis - eval_gt_vis, dim=-1).detach().cpu().numpy()
                eval_dict['joints3d_vis'].append(joints3d_vis)
            else:
                eval_dict['joints3d_vis'].append(np.zeros((0)))

            # occluded absolute data 
            if num_occl_pts > 0:
                eval_pred_occl = eval_pred_joints[occlusion_mask].reshape((num_occl_pts, 3))
                eval_gt_occl = eval_gt_joints[occlusion_mask].reshape((num_occl_pts, 3))
                joints3d_occl = torch.norm(eval_pred_occl - eval_gt_occl, dim=-1).detach().cpu().numpy()
                eval_dict['joints3d_occ'].append(joints3d_occl)
            else:
                eval_dict['joints3d_occ'].append(np.zeros((0)))

            # occl/vis aligned
            # visible absolute data
            if num_vis_pts > 0:
                eval_pred_align_vis = eval_pred_joints_align[vis_mask].reshape((num_vis_pts, 3))
                eval_gt_align_vis = eval_gt_joints_align[vis_mask].reshape((num_vis_pts, 3))
                joints3d_align_vis = torch.norm(eval_pred_align_vis - eval_gt_align_vis, dim=-1).detach().cpu().numpy()
                eval_dict['joints3d_align_vis'].append(joints3d_align_vis)
            else:
                eval_dict['joints3d_align_vis'].append(np.zeros((0)))

            # occluded absolute data 
            if num_occl_pts > 0:
                eval_pred_align_occl = eval_pred_joints_align[occlusion_mask].reshape((num_occl_pts, 3))
                eval_gt_align_occl = eval_gt_joints_align[occlusion_mask].reshape((num_occl_pts, 3))
                joints3d_align_occl = torch.norm(eval_pred_align_occl - eval_gt_align_occl, dim=-1).detach().cpu().numpy()
                eval_dict['joints3d_align_occ'].append(joints3d_align_occl)
            else:
                eval_dict['joints3d_align_occ'].append(np.zeros((0)))

    # per-joint acceleration
    _, pred_joint_accel_mag = compute_joint_accel(pred_joints_smpl)
    eval_dict['accel_mag'].append(pred_joint_accel_mag.detach().cpu().numpy())

    pred_joints_align = pred_joints_smpl - pred_joints_smpl[:,0:1,:]
    _, pred_joint_align_accel_mag = compute_joint_accel(pred_joints_align)
    eval_dict['accel_mag_align'].append(pred_joint_align_accel_mag.detach().cpu().numpy())

    # toe-floor penetration
    num_pen_list, num_tot, pen_dist = compute_toe_floor_pen(pred_joints_smpl, floor_plane, thresh_list=GRND_PEN_THRESH_LIST)
    eval_dict['ground_pen_dist'].append(pen_dist.detach().cpu().numpy())
    for thresh_idx, pen_thresh in enumerate(GRND_PEN_THRESH_LIST):
        cur_pen_key = get_grnd_pen_key(pen_thresh)
        eval_dict[cur_pen_key].append(num_pen_list[thresh_idx].detach().cpu().item())
        eval_dict[cur_pen_key + '_cnt'].append(num_tot)

    return  


def compute_subset_smpl_joint_err(eval_pred_joints, eval_gt_joints, subset='ee'):
    '''
    Compute SMPL joint position errors betwen pred_joints and gt_joints for the given subject.
    Assumed size of B x 22 x 3
    Options:
    - ee : end-effectors. hands, toebase, and ankle
    - legs : knees, angles, and toes
    '''
    subset_inds = None
    if subset == 'ee':
        subset_inds = [SMPL_JOINTS['leftFoot'], SMPL_JOINTS['rightFoot'],
                       SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase'],
                       SMPL_JOINTS['leftHand'], SMPL_JOINTS['rightHand']]
    elif subset == 'legs':
        subset_inds = [SMPL_JOINTS['leftFoot'], SMPL_JOINTS['rightFoot'],
                       SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase'],
                       SMPL_JOINTS['leftLeg'], SMPL_JOINTS['rightLeg']]
    else:
        print('Unrecognized joint subset!')
        exit()
    
    joint_err = torch.norm(eval_pred_joints[:, subset_inds] - eval_gt_joints[:, subset_inds], dim=-1)
    return joint_err

def compute_subset_comp_joint_err(eval_pred_joints, eval_gt_joints, subset='ee'):
    '''
    Compute comparison skeleton joint position errors betwen pred_joints and gt_joints for the given subject.
    Assumed size of B x J x 3
    Options:
    - ee : hands and ankels
    - legs : knees and ankles
    '''
    subset_inds = None
    if subset == 'ee':
        subset_inds = [COMP_JOINTS['RANK'], COMP_JOINTS['LANK'],
                       COMP_JOINTS['RWRI'], COMP_JOINTS['LWRI']]
    elif subset == 'legs':
        subset_inds = [COMP_JOINTS['RANK'], COMP_JOINTS['LANK'],
                       COMP_JOINTS['RKNE'], COMP_JOINTS['LKNE']]
    else:
        print('Unrecognized joint subset!')
        exit()
    
    joint_err = torch.norm(eval_pred_joints[:, subset_inds] - eval_gt_joints[:, subset_inds], dim=-1)
    return joint_err

def compute_joint_accel(joint_seq):
    ''' Magnitude of joint accelerations for joint_seq : T x J x 3 '''
    joint_accel = joint_seq[:-2] - (2*joint_seq[1:-1]) + joint_seq[2:]
    joint_accel = joint_accel / ((DATA_h**2))
    joint_accel_mag = torch.norm(joint_accel, dim=-1)
    return joint_accel, joint_accel_mag

def compute_toe_floor_pen(joint_seq, floor_plane, thresh_list=[0.0]):
    '''
    Given SMPL body joints sequence and the floor plane, computes number of times
    the toes penetrate the floor and the total number of frames.

    - thresh_list : compute the penetration ratio for each threshold in cm in this list

    Returns:
    - list of num_penetrations for each threshold, the number of total frames, and penetration distance at threshold 0.0
    '''
    toe_joints = joint_seq[:,[SMPL_JOINTS['leftToeBase'], SMPL_JOINTS['rightToeBase']], :]
    toe_joints = toe_joints.reshape((-1, 3))
    floor_normal = floor_plane[:3].reshape((1, 3))
    floor_normal = floor_normal / torch.norm(floor_normal, dim=-1, keepdim=True)
    floor_normal = floor_normal.expand_as(toe_joints)

    _, s = compute_plane_intersection(toe_joints, -floor_normal, floor_plane.reshape((1, 4)).expand((toe_joints.size(0), 4)))

    num_pen_list = torch.zeros((len(thresh_list))).to(torch.int).to(joint_seq.device)
    for thresh_idx, pen_thresh in enumerate(thresh_list):
        num_pen_thresh = torch.sum(s < -pen_thresh)
        num_pen_list[thresh_idx] = num_pen_thresh

    num_tot = s.size(0)

    pen_dist = torch.Tensor(np.array((0)))
    if torch.sum(s < 0) > 0:
        pen_dist = -s[s < 0] # distance of penetration at threshold of 0

    return num_pen_list, num_tot, pen_dist

# map from imapper gt 3d joints to comparison 12-joint skeleton
IMAP2COMPARE = [0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15] # no hips, neck, or head
COMP_ROOT_IDX = 4

IMAP_JOINTS = { 'RANK' : 0,  'RKNE' : 1, 'RHIP' : 2, 'LHIP' : 3,  'LKNE' : 4, 'LANK' : 5, 'PELV' : 6,
                 'THRX' : 7, 'NECK' : 8, 'HEAD' : 9, 'RWRI' : 10, 'RELB' : 11, 'RSHO' : 12,
                 'LSHO' : 13, 'LELB' : 14, 'LWRI' : 15}
IMAP_ID2NAME = {v : k for k, v in IMAP_JOINTS.items()}
COMP_NAMES = [IMAP_ID2NAME[i] for i in IMAP2COMPARE]
COMP_JOINTS = {jname : idx for idx, jname in enumerate(COMP_NAMES)}

# map from smpl regressed 3d joints to comparison 12-joint skeleton
SMPL2COMPARE = [ SMPL_JOINTS['rightFoot'], SMPL_JOINTS['rightLeg'], SMPL_JOINTS['leftLeg'], SMPL_JOINTS['leftFoot'],
                 SMPL_JOINTS['hips'], SMPL_JOINTS['neck'], SMPL_JOINTS['rightHand'],
                 SMPL_JOINTS['rightForeArm'], SMPL_JOINTS['rightArm'],  SMPL_JOINTS['leftArm'], SMPL_JOINTS['leftForeArm'],
                 SMPL_JOINTS['leftHand'] ]