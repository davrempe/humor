
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import shutil, glob
import os.path as osp
import cv2
import numpy as np
import json
import torch

from body_model.body_model import BodyModel

from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues, convert_to_rotmat
from utils.logging import mkdir, Logger

NSTAGES = 3 # number of stages in the optimization
DEFAULT_FOCAL_LEN = (1060.531764702488, 1060.3856705041237) # fx, fy

def read_keypoints(keypoint_fn):
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float)

    person_data = data['people'][0]
    body_keypoints = np.array(person_data['pose_keypoints_2d'],
                                dtype=np.float)
    body_keypoints = body_keypoints.reshape([-1, 3])

    return body_keypoints

def resize_points(points_arr, num_pts):
    '''
    Either randomly subsamples or pads the given points_arr to be of the desired size.
    - points_arr : N x 3
    - num_pts : desired num point 
    '''
    is_torch = isinstance(points_arr, torch.Tensor)
    N = points_arr.size(0) if is_torch else points_arr.shape[0]
    if N > num_pts:
        samp_inds = np.random.choice(np.arange(N), size=num_pts, replace=False)
        points_arr = points_arr[samp_inds]
    elif N < num_pts:
        while N < num_pts:
            pad_size = num_pts - N
            if is_torch:
                points_arr = torch.cat([points_arr, points_arr[:pad_size]], dim=0)
                N = points_arr.size(0)
            else:
                points_arr = np.concatenate([points_arr, points_arr[:pad_size]], axis=0)
                N = points_arr.shape[0]
    return points_arr

def compute_plane_intersection(point, direction, plane):
    '''
    Given a ray defined by a point in space and a direction, compute the intersection point with the given plane.
    Detect intersection in either direction or -direction so the given ray may not actually intersect with the plane.

    Returns the intersection point as well as s such that point + s*direction = intersection_point. if s < 0 it means
    -direction intersects.

    - point : B x 3
    - direction : B x 3
    - plane : B x 4 (a, b, c, d) where (a, b, c) is the normal and (d) the offset.
    '''
    plane_normal = plane[:,:3]
    plane_off = plane[:,3]
    s = (plane_off - bdot(plane_normal, point)) / bdot(plane_normal, direction)
    itsct_pt = point + s.reshape((-1, 1))*direction
    return itsct_pt, s

def bdot(A1, A2, keepdim=False):
    ''' 
    Batched dot product.
    - A1 : B x D
    - A2 : B x D.
    Returns B.
    '''
    return (A1*A2).sum(dim=-1, keepdim=keepdim) 

def parse_floor_plane(floor_plane):
    '''
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    '''
    floor_offset = torch.norm(floor_plane, dim=1, keepdim=True)
    floor_normal = floor_plane / floor_offset
    
    # in camera system -y is up, so floor plane normal y component should never be positive
    #       (assuming the camera is not sideways or upside down)
    neg_mask = floor_normal[:,1:2] > 0.0
    floor_normal = torch.where(neg_mask.expand_as(floor_normal), -floor_normal, floor_normal)
    floor_offset = torch.where(neg_mask, -floor_offset, floor_offset)
    floor_plane_4d = torch.cat([floor_normal, floor_offset], dim=1)

    return floor_plane_4d

def load_planercnn_res(res_path):
    '''
    Given a directory containing PlaneRCNN plane detection results, loads the first image result 
    and heuristically finds and returns the floor plane.
    '''
    planes_param_path = glob.glob(res_path + '/*_plane_parameters_*.npy')[0]
    planes_mask_path = glob.glob(res_path + '/*_plane_masks_*.npy')[0]
    planes_params = np.load(planes_param_path)
    planes_masks = np.load(planes_mask_path)
    
    # heuristically determine the ground plane
    #   the plane with the most labeled pixels in the bottom N rows
    nrows = 10
    label_count = np.sum(planes_masks[:, -nrows:, :], axis=(1, 2))
    floor_idx = np.argmax(label_count)
    valid_floor = False
    floor_plane = None
    while not valid_floor:
        # loop until we find a plane with many pixels on the bottom
        #       and doesn't face in the complete wrong direction
        # we assume the y component is larger than any others
        # i.e. that the floor is not > 45 degrees relative rotation from the camera
        floor_plane = planes_params[floor_idx]
        # transform to our system
        floor_plane = np.array([floor_plane[0], -floor_plane[2], floor_plane[1]])
        # determine 4D parameterization
        # for this data we know y should always be negative
        floor_offset = np.linalg.norm(floor_plane)
        floor_normal = floor_plane / floor_offset
        if floor_normal[1] > 0.0:
            floor_offset *= -1.0
            floor_normal *= -1.0
        a, b, c = floor_normal
        d = floor_offset
        floor_plane = np.array([a, b, c, d])

        valid_floor = np.abs(b) > np.abs(a) and np.abs(b) > np.abs(c)
        if not valid_floor:
            label_count[floor_idx] = 0
            floor_idx = np.argmax(label_count)

    return floor_plane


def compute_cam2prior(floor_plane, trans, root_orient, joints):
    '''
    Computes rotation and translation from the camera frame to the canonical coordinate system
    used by the motion and initial state priors.
    - floor_plane : B x 3
    - trans : B x 3
    - root_orient : B x 3
    - joints : B x J x 3
    '''
    B = floor_plane.size(0)
    if floor_plane.size(1) == 3:
        floor_plane_4d = parse_floor_plane(floor_plane)
    else:
        floor_plane_4d = floor_plane
    floor_normal = floor_plane_4d[:,:3]
    floor_trans, _ = compute_plane_intersection(trans, -floor_normal, floor_plane_4d)

    # compute prior frame axes within the camera frame
    # up is the floor_plane normal
    up_axis = floor_normal
    # right is body -x direction projected to floor plane
    root_orient_mat = batch_rodrigues(root_orient)
    body_right = -root_orient_mat[:, :, 0]
    floor_body_right, s = compute_plane_intersection(trans, body_right, floor_plane_4d)
    right_axis = floor_body_right - floor_trans 
    # body right may not actually intersect - in this case must negate axis because we have the -x
    right_axis = torch.where(s.reshape((B, 1)) < 0, -right_axis, right_axis)
    right_axis = right_axis / torch.norm(right_axis, dim=1, keepdim=True)
    # forward is their cross product
    fwd_axis = torch.cross(up_axis, right_axis)
    fwd_axis = fwd_axis / torch.norm(fwd_axis, dim=1, keepdim=True)

    prior_R = torch.stack([right_axis, fwd_axis, up_axis], dim=2)
    cam2prior_R = prior_R.transpose(2, 1)

    # translation takes translation to origin plus offset to the floor
    cam2prior_t = -trans

    _, s_root = compute_plane_intersection(joints[:,0], -floor_normal, floor_plane_4d)
    root_height = s_root.reshape((B, 1))

    return cam2prior_R, cam2prior_t, root_height

def apply_robust_weighting(res, robust_loss_type='bisquare', robust_tuning_const=4.6851):
    '''
    Returns robustly weighted squared residuals.
    - res : torch.Tensor (B x N), take the MAD over each batch dimension independently.
    '''
    robust_choices = ['none', 'bisquare']
    if robust_loss_type not in robust_choices:
        print('Not a valid robust loss: %s. Please use %s' % (robust_loss_type, str(robust_choices)))
    
    w = None
    detach_res = res.clone().detach() # don't want gradients flowing through the weights to avoid degeneracy
    if robust_loss_type == 'none':
        w = torch.ones_like(detach_res)
    elif robust_loss_type == 'bisquare':
        w = bisquare_robust_weights(detach_res, tune_const=robust_tuning_const)

    # apply weights to squared residuals
    weighted_sqr_res = w * (res**2)
    return weighted_sqr_res, w

def robust_std(res):
    ''' 
    Compute robust estimate of standarad deviation using median absolute deviation (MAD)
    of the given residuals independently over each batch dimension.

    - res : (B x N)

    Returns:
    - std : B x 1
    '''
    B = res.size(0)
    med = torch.median(res, dim=-1)[0].reshape((B,1))
    abs_dev = torch.abs(res - med)
    MAD = torch.median(abs_dev, dim=-1)[0].reshape((B, 1))
    std = MAD / 0.67449
    return std

def bisquare_robust_weights(res, tune_const=4.6851):
    '''
    Bisquare (Tukey) loss.
    See https://www.mathworks.com/help/curvefit/least-squares-fitting.html

    - residuals
    '''
    # print(res.size())
    norm_res = res / (robust_std(res) * tune_const)
    # NOTE: this should use absolute value, it's ok right now since only used for 3d point cloud residuals
        #   which are guaranteed positive, but generally this won't work)
    outlier_mask = norm_res >= 1.0

    # print(torch.sum(outlier_mask))
    # print('Outlier frac: %f' % (float(torch.sum(outlier_mask)) / res.size(1)))

    w = (1.0 - norm_res**2)**2
    w[outlier_mask] = 0.0

    return w

def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def log_cur_stats(stats_dict, loss, iter=None):
    Logger.log('LOSS: %f' % (loss.cpu().item()))
    print('----')
    for k, v in stats_dict.items():
        if isinstance(v, float):
            Logger.log('%s: %f' % (k, v))
        else:
            Logger.log('%s: %f' % (k, v.cpu().item()))
    if iter is not None:
        print('======= iter %d =======' % (int(iter)))
    else:
        print('========')

def save_optim_result(cur_res_out_paths, optim_result, per_stage_results, gt_data, observed_data, data_type,
                      optim_floor=True,
                      obs_img_paths=None,
                      obs_mask_paths=None):
    # final optim results
    res_betas = optim_result['betas'].cpu().numpy()
    res_trans = optim_result['trans'].cpu().numpy()
    res_root_orient = optim_result['root_orient'].cpu().numpy()
    res_body_pose = optim_result['pose_body'].cpu().numpy()
    res_contacts = None
    res_floor_plane = None
    if 'contacts' in optim_result:
        res_contacts = optim_result['contacts'].cpu().numpy()
    if 'floor_plane' in optim_result:
        res_floor_plane = optim_result['floor_plane'].cpu().numpy()
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        cur_res_out_path = os.path.join(cur_res_out_path, 'stage3_results.npz')
        save_dict = { 
            'betas' : res_betas[bidx],
            'trans' : res_trans[bidx],
            'root_orient' : res_root_orient[bidx],
            'pose_body' : res_body_pose[bidx]
        }
        if res_contacts is not None:
            save_dict['contacts'] = res_contacts[bidx]
        if res_floor_plane is not None:
            save_dict['floor_plane'] = res_floor_plane[bidx]
        np.savez(cur_res_out_path, **save_dict)

    # in prior coordinate frame
    if 'stage3' in per_stage_results and optim_floor:
        res_trans = per_stage_results['stage3']['prior_trans'].detach().cpu().numpy()
        res_root_orient = per_stage_results['stage3']['prior_root_orient'].detach().cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'stage3_results_prior.npz')
            save_dict = { 
                'betas' : res_betas[bidx],
                'trans' : res_trans[bidx],
                'root_orient' : res_root_orient[bidx],
                'pose_body' : res_body_pose[bidx]
            }
            if res_contacts is not None:
                save_dict['contacts'] = res_contacts[bidx]
            np.savez(cur_res_out_path, **save_dict)

    # ground truth
    save_gt = 'betas' in gt_data and \
                'trans' in gt_data and \
                'root_orient' in gt_data and \
                'pose_body' in gt_data
    if save_gt:
        gt_betas = gt_data['betas'].cpu().numpy()
        if data_type not in ['PROX-RGB', 'PROX-RGBD']:
            gt_betas = gt_betas[:,0] # only need frame 1 for e.g. 3d data since it's the same over time.
        gt_trans = gt_data['trans'].cpu().numpy()
        gt_root_orient = gt_data['root_orient'].cpu().numpy()
        gt_body_pose = gt_data['pose_body'].cpu().numpy()
        gt_contacts = None
        if 'contacts' in gt_data:
            gt_contacts = gt_data['contacts'].cpu().numpy()
        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            gt_res_name = 'proxd_results.npz' if data_type in ['PROX-RGB', 'PROX-RGBD'] else 'gt_results.npz'
            cur_gt_out_path = os.path.join(cur_res_out_path, gt_res_name)
            save_dict = { 
                'betas' : gt_betas[bidx],
                'trans' : gt_trans[bidx],
                'root_orient' : gt_root_orient[bidx],
                'pose_body' : gt_body_pose[bidx]
            }
            if gt_contacts is not None:
                save_dict['contacts'] = gt_contacts[bidx]
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            np.savez(cur_gt_out_path, **save_dict)

            # if these are proxd results also need to save a GT with cam matrix
            if data_type in ['PROX-RGB', 'PROX-RGBD']:
                cur_gt_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
                np.savez(cur_gt_out_path, cam_mtx=cam_mat[bidx])

    elif 'joints3d' in gt_data:
        # don't have smpl params, but have 3D joints (e.g. imapper)
        gt_joints = gt_data['joints3d'].cpu().numpy()
        cam_mat = occlusions = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].cpu().numpy()
        if 'occlusions' in gt_data:
            occlusions = gt_data['occlusions'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'joints3d' : gt_joints[bidx]
            }
            if cam_mat is not None:
                save_dict['cam_mtx'] = cam_mat[bidx]
            if occlusions is not None:
                save_dict['occlusions'] = occlusions[bidx]
            np.savez(cur_res_out_path, **save_dict)
    elif 'cam_matx' in gt_data:
        # need the intrinsics even if we have nothing else
        cam_mat = gt_data['cam_matx'].cpu().numpy()
        for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
            cur_res_out_path = os.path.join(cur_res_out_path, 'gt_results.npz')
            save_dict = { 
                'cam_mtx' : cam_mat[bidx]
            }
            np.savez(cur_res_out_path, **save_dict)

    # observations
    obs_out = {k : v.cpu().numpy() for k, v in observed_data.items() if k != 'prev_batch_overlap_res'}
    for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
        obs_out_path = os.path.join(cur_res_out_path, 'observations.npz')
        cur_obs_out = {k : v[bidx] for k, v in obs_out.items() if k not in ['RGB']}
        if obs_img_paths is not None:
            cur_obs_out['img_paths'] = [frame_tup[bidx] for frame_tup in obs_img_paths]
            # print(cur_obs_out['img_paths'])
        if obs_mask_paths is not None:
            cur_obs_out['mask_paths'] = [frame_tup[bidx] for frame_tup in obs_mask_paths]
        np.savez(obs_out_path, **cur_obs_out)    


def save_rgb_stitched_result(seq_intervals, all_res_out_paths, res_out_path, device,
                                body_model_path, num_betas, use_joints2d):
    import cv2
    seq_overlaps = [0]
    for int_idx in range(len(seq_intervals)-1):
        prev_end = seq_intervals[int_idx][1]
        cur_start = seq_intervals[int_idx+1][0]
        seq_overlaps.append(prev_end - cur_start)

    # if arbitray RGB video data, stitch together to save full sequence output
    all_res_dirs = all_res_out_paths
    print(all_res_dirs)

    final_res_out_path = os.path.join(res_out_path, 'final_results')
    mkdir(final_res_out_path)

    concat_cam_res = None
    concat_contacts = None
    concat_ground_planes = None
    concat_joints2d = None
    concat_img_paths = None
    gt_cam_mtx = None
    for res_idx, res_dir in enumerate(all_res_dirs):
        # camera view
        cur_stage3_res = load_res(res_dir, 'stage3_results.npz')
        cur_contacts = torch.Tensor(cur_stage3_res['contacts']).to(device)
        if concat_ground_planes is None: 
            concat_ground_planes = torch.Tensor(cur_stage3_res['floor_plane']).to(device).reshape((1, -1))
        else:
            concat_ground_planes = torch.cat([concat_ground_planes, torch.Tensor(cur_stage3_res['floor_plane']).to(device).reshape((1, -1))], dim=0)
        cur_stage3_res = {k : v for k, v in cur_stage3_res.items() if k in ['betas', 'trans', 'root_orient', 'pose_body']}
        cur_stage3_res = prep_res(cur_stage3_res, device, cur_stage3_res['trans'].shape[0])
        if concat_cam_res is None: 
            concat_cam_res = cur_stage3_res
            concat_contacts = cur_contacts
        else:
            for k, v in concat_cam_res.items():
                concat_cam_res[k] = torch.cat([concat_cam_res[k], cur_stage3_res[k][seq_overlaps[res_idx]:]], dim=0)
            concat_contacts = torch.cat([concat_contacts, cur_contacts[seq_overlaps[res_idx]:]], dim=0)

        # gt
        if gt_cam_mtx is None:
            gt_res = load_res(res_dir, 'gt_results.npz')
            gt_cam_mtx = gt_res['cam_mtx']

        # obs
        cur_obs = load_res(res_dir, 'observations.npz')
        if concat_joints2d is None:
            concat_joints2d = cur_obs['joints2d']
        else:
            concat_joints2d = np.concatenate([concat_joints2d, cur_obs['joints2d'][seq_overlaps[res_idx]:]], axis=0)
        if concat_img_paths is None:
            concat_img_paths = list(cur_obs['img_paths'])
        else:
            concat_img_paths = concat_img_paths + list(cur_obs['img_paths'][seq_overlaps[res_idx]:])
        
        # ignore if we don't have an interval for this directory (was an extra due to even batching requirement)
        if res_idx >= len(seq_overlaps):
            break

    # copy meta
    src_meta_path = os.path.join(all_res_dirs[0], 'meta.txt')
    shutil.copyfile(src_meta_path, os.path.join(final_res_out_path, 'meta.txt'))

    #  gt results (cam matx)
    np.savez(os.path.join(final_res_out_path, 'gt_results.npz'), cam_mtx=gt_cam_mtx)

    # obs results (joints2d and img_paths)
    np.savez(os.path.join(final_res_out_path, 'observations.npz'), joints2d=concat_joints2d, img_paths=concat_img_paths)

    #  save the actual results npz for viz later
    concat_res_out_path = os.path.join(final_res_out_path, 'stage3_results.npz')
    res_betas = concat_cam_res['betas'].clone().detach().cpu().numpy()
    res_trans = concat_cam_res['trans'].clone().detach().cpu().numpy()
    res_root_orient = concat_cam_res['root_orient'].clone().detach().cpu().numpy()
    res_body_pose = concat_cam_res['pose_body'].clone().detach().cpu().numpy()
    res_floor_plane = concat_ground_planes[0].clone().detach().cpu().numpy() # NOTE: saves estimate from first subsequence
    res_contacts = concat_contacts.clone().detach().cpu().numpy()
    np.savez(concat_res_out_path, betas=res_betas,
                                trans=res_trans,
                                root_orient=res_root_orient,
                                pose_body=res_body_pose,
                                floor_plane=res_floor_plane,
                                contacts=res_contacts)

    # get body model
    num_viz_frames = concat_cam_res['trans'].size(0)
    viz_body_model = BodyModel(bm_path=body_model_path,
                            num_betas=num_betas,
                            batch_size=num_viz_frames,
                            use_vtx_selector=use_joints2d).to(device)
    viz_body = run_smpl(concat_cam_res, viz_body_model)
    
    # transform full camera-frame sequence into a shared prior frame based on a single ground plane
    viz_joints3d = viz_body.Jtr
    # compute the transformation based on t=0 and the first sequence floor plane
    cam2prior_R, cam2prior_t, cam2prior_root_height = compute_cam2prior(concat_ground_planes[0].unsqueeze(0),
                                                                        concat_cam_res['trans'][0].unsqueeze(0),
                                                                        concat_cam_res['root_orient'][0].unsqueeze(0),
                                                                        viz_joints3d[0].unsqueeze(0))
    # transform the whole sequence
    input_data_dict = {kb : vb.unsqueeze(0) for kb, vb in concat_cam_res.items() if kb in ['trans', 'root_orient', 'pose_body', 'betas']}
    viz_prior_data_dict = apply_cam2prior(input_data_dict, cam2prior_R, cam2prior_t, cam2prior_root_height, 
                                            input_data_dict['pose_body'],
                                            input_data_dict['betas'],
                                            0,
                                            viz_body_model)
    concat_prior_res = {
        'trans' : viz_prior_data_dict['trans'][0],
        'root_orient' : viz_prior_data_dict['root_orient'][0],
        'pose_body' : concat_cam_res['pose_body'],
        'betas' : concat_cam_res['betas']
    }

    # save pose prior frame
    concat_prior_res_out_path = os.path.join(final_res_out_path, 'stage3_results_prior.npz')
    res_betas = concat_prior_res['betas'].clone().detach().cpu().numpy()
    res_trans = concat_prior_res['trans'].clone().detach().cpu().numpy()
    res_root_orient = concat_prior_res['root_orient'].clone().detach().cpu().numpy()
    res_body_pose = concat_prior_res['pose_body'].clone().detach().cpu().numpy()
    res_contacts = concat_contacts.clone().detach().cpu().numpy()
    np.savez(concat_prior_res_out_path, betas=res_betas,
                                trans=res_trans,
                                root_orient=res_root_orient,
                                pose_body=res_body_pose,
                                contacts=res_contacts)


def load_res(result_dir, file_name):
    '''
    Load np result from our model or GT
    '''
    res_path = os.path.join(result_dir, file_name)
    if not os.path.exists(res_path):
        return None
    res = np.load(res_path)
    res_dict = {k : res[k] for k in res.files}
    return res_dict

def prep_res(np_res, device, T):
    '''
    Load np result dict into dict of torch objects for use with SMPL body model.
    '''
    betas = np_res['betas']
    betas = torch.Tensor(betas).to(device)
    if len(betas.size()) == 1:
        num_betas = betas.size(0)
        betas = betas.reshape((1, num_betas)).expand((T, num_betas))
    else:
        num_betas = betas.size(1)
        assert(betas.size(0) == T)
    trans = np_res['trans']
    trans = torch.Tensor(trans).to(device)
    root_orient = np_res['root_orient']
    root_orient = torch.Tensor(root_orient).to(device)
    pose_body = np_res['pose_body']
    pose_body = torch.Tensor(pose_body).to(device)

    res_dict = {
        'betas' : betas,
        'trans' : trans,
        'root_orient' : root_orient,
        'pose_body' : pose_body
    }

    for k, v in np_res.items():
        if k not in ['betas', 'trans', 'root_orient', 'pose_body']:
            res_dict[k] = v
    return res_dict

def run_smpl(res_dict, body_model):
    smpl_body = body_model(pose_body=res_dict['pose_body'], 
                            pose_hand=None, 
                            betas=res_dict['betas'],
                            root_orient=res_dict['root_orient'],
                            trans=res_dict['trans'])
    return smpl_body

def apply_cam2prior(data_dict, R, t, root_height, body_pose, betas, key_frame_idx, body_model, inverse=False):
    '''
    Applies the camera2prior tranformation made up of R, t to the data in data dict and
    returns a new dictionary with the transformed data.
    Right now supports: trans, root_orient.

    NOTE: If the number of timesteps in trans/root_orient is 1, this function assumes they are at key_frame_idx.
            (othherwise the calculation of cur_root_height or trans_offset in inverse case is not correct)

    key_frame_idx : the timestep used to compute cam2prior size (B) tensor
    inverse : if true, applies the inverse transformation from prior space to camera
    '''
    prior_dict = dict()
    if 'root_orient' in data_dict:
        # B x T x 3
        root_orient = data_dict['root_orient']
        B, T, _ = root_orient.size()
        R_time = R.unsqueeze(1).expand((B, T, 3, 3))
        t_time = t.unsqueeze(1).expand((B, T, 3))
        root_orient_mat = batch_rodrigues(root_orient.reshape((-1, 3))).reshape((B, T, 3, 3))
        if inverse:
            prior_root_orient_mat = torch.matmul(R_time.transpose(3, 2), root_orient_mat)
        else:
            prior_root_orient_mat = torch.matmul(R_time, root_orient_mat)
        prior_root_orient = rotation_matrix_to_angle_axis(prior_root_orient_mat.reshape((B*T, 3, 3))).reshape((B, T, 3))
        prior_dict['root_orient'] = prior_root_orient

    if 'trans' in data_dict and 'root_orient' in data_dict:
        # B x T x 3
        trans = data_dict['trans']
        B, T, _ = trans.size()
        R_time = R.unsqueeze(1).expand((B, T, 3, 3))
        t_time = t.unsqueeze(1).expand((B, T, 3))
        if inverse:
            # transform so key frame at origin
            if T > 1:
                trans_offset = trans[np.arange(B),key_frame_idx,:].unsqueeze(1)
            else:
                trans_offset = trans[:,0:1,:]
            trans = trans - trans_offset
            # rotates to camera frame
            trans = torch.matmul(R_time.transpose(3, 2), trans.reshape((B, T, 3, 1)))[:,:,:,0]
            # translate to camera frame
            trans = trans - t_time
        else:
            # first transform so the trans of key frame is at origin
            trans = trans + t_time
            # then rotate to canonical frame
            trans = torch.matmul(R_time, trans.reshape((B, T, 3, 1)))[:,:,:,0]
            # then apply floor offset so the root joint is at the desired height
            cur_smpl_body = body_model(pose_body=body_pose.reshape((-1, body_pose.size(2))), 
                                    pose_hand=None, 
                                    betas=betas.reshape((-1, betas.size(2))),
                                    root_orient=prior_dict['root_orient'].reshape((-1, 3)),
                                    trans=trans.reshape((-1, 3)))
            smpl_joints3d = cur_smpl_body.Jtr.reshape((B, T, -1, 3))
            if T > 1:
                cur_root_height = smpl_joints3d[np.arange(B),key_frame_idx,0,2:3]
            else:
                cur_root_height = smpl_joints3d[:,0,0,2:3]
            height_diff = root_height - cur_root_height
            trans_offset = torch.cat([torch.zeros((B, 2)).to(height_diff), height_diff], axis=1)
            trans = trans + trans_offset.reshape((B, 1, 3))
        prior_dict['trans'] = trans
    elif 'trans' in data_dict:
        Logger.log('Cannot apply cam2prior on translation without root orient data!')
        exit()

    return prior_dict


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    Adapted from https://github.com/mkocabas/VIBE/blob/master/lib/models/spin.py
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, 2): Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

OP_NUM_JOINTS = 25
OP_IGNORE_JOINTS = [1, 9, 12] # neck and left/right hip
OP_EDGE_LIST = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18], [14,19], [19,20], [14,21], [11,22], [22,23], [11,24]]
# indices to map an openpose detection to its flipped version
OP_FLIP_MAP = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]

#
# The following 2 functions are borrowed from VPoser (https://github.com/nghorbani/human_body_prior).
# See their license for usage restrictions.
#
def expid2model(expr_dir):
    from configer import Configer

    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_fname = sorted(glob.glob(os.path.join(expr_dir, 'snapshots', '*.pt')), key=os.path.getmtime)[-1]
    try_num = os.path.basename(best_model_fname).split('_')[0]

    print(('Found Trained Model: %s' % best_model_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(
        default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_fname)

    return ps, best_model_fname

def load_vposer(expr_dir, vp_model='snapshot'):
    '''
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''
    import importlib
    import os
    import torch

    ps, trained_model_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':

        vposer_path = sorted(glob.glob(os.path.join(expr_dir, 'vposer_*.py')), key=os.path.getmtime)[-1]

        spec = importlib.util.spec_from_file_location('VPoser', vposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        vposer_pt = getattr(module, 'VPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        vposer_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    vposer_pt.load_state_dict(torch.load(trained_model_fname, map_location='cpu'))
    vposer_pt.eval()

    return vposer_pt, ps