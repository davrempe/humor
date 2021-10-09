'''
Test-time optimization to fit to observations using HuMoR as a motion prior.

This is a 3-stage optimization. Stages 1 & 2 are initialization that DON'T use the motion prior,
and stage 3 is the main optimization that uses HuMoR.
'''


import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time, math, shutil, json

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from utils.logging import Logger, cp_files

from models.humor_model import HumorModel
from datasets.amass_discrete_dataset import AmassDiscreteDataset
from datasets.amass_fit_dataset import AMASSFitDataset
from datasets.prox_dataset import ProxDataset
from datasets.imapper_dataset import iMapperDataset
from datasets.rgb_dataset import RGBVideoDataset
from utils.torch import load_state
from utils.logging import mkdir
from fitting.config import parse_args
from fitting.fitting_utils import NSTAGES, DEFAULT_FOCAL_LEN, load_vposer, save_optim_result, save_rgb_stitched_result
from fitting.motion_optimizer import MotionOptimizer
from utils.video import video_to_images, run_openpose, run_deeplab_v3

from body_model.body_model import BodyModel
from body_model.utils import SMPLX_PATH, SMPLH_PATH

def main(args, config_file):
    res_out_path = None
    if args.out is not None:
        mkdir(args.out)
        # create logging system
        fit_log_path = os.path.join(args.out, 'fit_' + str(int(time.time())) + '.log')
        Logger.init(fit_log_path)

        if args.save_results or args.save_stages_results:
            res_out_path = os.path.join(args.out, 'results_out')

    # save arguments used
    Logger.log('args: ' + str(args))
    # and save config
    cp_files(args.out, [config_file])
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    B = args.batch_size
    if args.amass_batch_size > 0:
        B = args.amass_batch_size
    if args.prox_batch_size > 0:
        B = args.prox_batch_size
    if B == 3:
        Logger.log('Cannot use batch size 3, setting to 2!') # NOTE: bug with pytorch 3x3 matmul weirdness
        B = 2
    dataset = None
    data_fps = args.data_fps
    im_dim = (1080, 1080)
    rgb_img_folder = None
    rgb_vid_name = None
    if args.data_type == 'AMASS':
        dataset = AMASSFitDataset(args.data_path,
                                  seq_len=args.amass_seq_len,
                                  return_joints=args.amass_use_joints,
                                  return_verts=args.amass_use_verts,
                                  return_points=args.amass_use_points,
                                  noise_std=args.amass_noise_std,
                                  make_partial=args.amass_make_partial,
                                  partial_height=args.amass_partial_height,
                                  drop_middle=args.amass_drop_middle,
                                  root_only=args.amass_root_joint_only,
                                  split_by=args.amass_split_by,
                                  custom_split=args.amass_custom_split)
        data_fps = 30
    elif args.data_type == 'PROX-RGB' or args.data_type == 'PROX-RGBD':
        dataset = ProxDataset(args.data_path,
                              quant=False,
                              split='test',
                              seq_len=args.prox_seq_len,
                              load_depth=(args.data_type == 'PROX-RGBD'),
                              max_pts=4096,
                              load_img=False,
                              recording=args.prox_recording,
                              recording_subseq_idx=args.prox_recording_subseq_idx,
                              return_fitting=True,
                              load_scene_mesh=False,
                              mask_color=True,
                              mask_joints=args.mask_joints2d,
                              return_mask=False,
                              estimate_floor_plane=False, # don't want to use GT floor
                              load_floor_plane=True, # use PlaneRCNN instead
                              flip=True # must be flipped if load_floor_plane only need this to match the scene/PROXD fittings for comparison
                            )
        data_fps = 30
        im_dim = (1920, 1080)
    elif args.data_type == 'iMapper-RGB':
        dataset = iMapperDataset(args.data_path,
                                seq_len=args.imapper_seq_len,
                                load_img=False,
                                load_floor_plane=True,
                                scene=args.imapper_scene,
                                scene_subseq_idx=args.imapper_scene_subseq_idx,
                                mask_joints=args.mask_joints2d
                            )
        data_fps = 30
        im_dim = (1920, 1080)
    elif args.data_type == 'RGB':
        # preprocess video to info needed for optim
        video_preprocess_path = os.path.join(args.out, 'rgb_preprocess')
        img_folder = os.path.join(video_preprocess_path, 'raw_frames')
        mask_out_path = None
        if args.mask_joints2d:
            mask_out_path = os.path.join(video_preprocess_path, 'masks')
        op_out_path = os.path.join(video_preprocess_path, 'op_keypoints')

        # if we already did precprocessing (ran sam video before), skip it
        if not os.path.exists(video_preprocess_path):
            mkdir(video_preprocess_path)
            # video -> images (at 30 Hz) - save to new output directory
            img_folder, _, img_shape = video_to_images(args.data_path,
                                                            fps=30,
                                                            img_folder=img_folder,
                                                            return_info=True)
            print(img_folder)
            print(img_shape)
            # OpenPose on images
            op_frames_out = os.path.join(video_preprocess_path, 'op_frames')
            run_openpose(args.openpose, img_folder,
                        op_out_path,
                        img_out=op_frames_out,
                        video_out=os.path.join(video_preprocess_path, 'op_keypoints_overlay.mp4'))

            # if desired segmentation mask on images
            if args.mask_joints2d:
                run_deeplab_v3(img_folder, img_shape[:2],
                            mask_out_path,
                            batch_size=4)
        else:
            print('Already pre-processed video, skipping pre-processing...')
            import cv2
            img_shape = cv2.imread(os.path.join(img_folder, '000001.png')).shape
            if args.mask_joints2d and not os.path.exists(mask_out_path):
                print('Could not find detected masks from previous pre-processing! Please delete rgb_preprocess directory and re-run!')
                exit()

        # read in intrinsics if given
        cam_mat = None
        if args.rgb_intrinsics is None:
            cam_mat = np.array([[DEFAULT_FOCAL_LEN[0], 0.0, img_shape[1] / 2.],
                                [0.0, DEFAULT_FOCAL_LEN[1], img_shape[0] / 2.],
                                [0.0, 0.0, 1.0]])
        else:
            with open(args.rgb_intrinsics, 'r') as f:
                intrins_data = json.load(f)
            cam_mat = np.array(intrins_data)
        
        # Create dataset by splitting the video up into overlapping clips
        vid_name = '.'.join(args.data_path.split('/')[-1].split('.')[:-1])
        dataset = RGBVideoDataset(op_out_path,
                                  cam_mat,
                                  seq_len=args.rgb_seq_len,
                                  overlap_len=args.rgb_overlap_len,
                                  img_path=img_folder,
                                  load_img=False,
                                  masks_path=mask_out_path,
                                  mask_joints=args.mask_joints2d,
                                  planercnn_path=args.rgb_planercnn_res,
                                  video_name=vid_name
                                )

        data_fps = 30
        im_dim = tuple(img_shape[:-1][::-1])
        rgb_img_folder = img_folder
        rgb_vid_name = vid_name
    else:
        raise NotImplementedError('Fitting for arbitrary RGB-D videos is not yet implemented')

    data_loader = DataLoader(dataset, 
                            batch_size=B,
                            shuffle=args.shuffle,
                            num_workers=0,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    # weights for optimization loss terms
    loss_weights = {
        'joints2d' : args.joint2d_weight,
        'joints3d' : args.joint3d_weight,
        'joints3d_rollout' : args.joint3d_rollout_weight,
        'verts3d' : args.vert3d_weight,
        'points3d' : args.point3d_weight,
        'pose_prior' : args.pose_prior_weight,
        'shape_prior' : args.shape_prior_weight,
        'motion_prior' : args.motion_prior_weight,
        'init_motion_prior' : args.init_motion_prior_weight,
        'joint_consistency' : args.joint_consistency_weight,
        'bone_length' : args.bone_length_weight,
        'joints3d_smooth' : args.joint3d_smooth_weight,
        'contact_vel' : args.contact_vel_weight,
        'contact_height' : args.contact_height_weight,
        'floor_reg' : args.floor_reg_weight,
        'rgb_overlap_consist' : args.rgb_overlap_consist_weight
    }

    max_loss_weights = {k : max(v) for k, v in loss_weights.items()}
    all_stage_loss_weights = []
    for sidx in range(NSTAGES):
        stage_loss_weights = {k : v[sidx] for k, v in loss_weights.items()}
        all_stage_loss_weights.append(stage_loss_weights)
        
    use_joints2d = max_loss_weights['joints2d'] > 0.0

    # must always have pose prior to optimize in latent space
    pose_prior, _ = load_vposer(args.vposer)
    pose_prior = pose_prior.to(device)
    pose_prior.eval()

    motion_prior = None
    Logger.log('Loading motion prior from %s...' % (args.humor))
    motion_prior = HumorModel(in_rot_rep=args.humor_in_rot_rep, 
                                out_rot_rep=args.humor_out_rot_rep,
                                latent_size=args.humor_latent_size,
                                model_data_config=args.humor_model_data_config,
                                steps_in=args.humor_steps_in)
    motion_prior.to(device)
    load_state(args.humor, motion_prior, map_location=device)
    motion_prior.eval()

    # load in prior on the initial motion state if given
    init_motion_prior = None
    if motion_prior is not None and max_loss_weights['init_motion_prior'] > 0.0:
        Logger.log('Loading initial motion state prior from %s...' % (args.init_motion_prior))
        gmm_path = os.path.join(args.init_motion_prior, 'prior_gmm.npz')
        init_motion_prior = dict()
        if os.path.exists(gmm_path):
            gmm_res = np.load(gmm_path)
            gmm_weights = torch.Tensor(gmm_res['weights']).to(device)
            gmm_means = torch.Tensor(gmm_res['means']).to(device)
            gmm_covs = torch.Tensor(gmm_res['covariances']).to(device)
            init_motion_prior['gmm'] = (gmm_weights, gmm_means, gmm_covs)
        if len(init_motion_prior.keys()) == 0:
            Logger.log('Could not find init motion state prior at given directory!')
            exit()

    fit_errs = dict()
    prev_batch_overlap_res_dict = None
    use_overlap_loss = sum(loss_weights['rgb_overlap_consist']) > 0.0
    for i, data in enumerate(data_loader):
        start_t = time.time()
        # these dicts have different data depending on modality
        # MUST have at least name
        observed_data, gt_data = data
        # both of these are a list of tuples, each list index is a frame and the tuple index is the seq within the batch
        obs_img_paths = None if 'img_paths' not in observed_data else observed_data['img_paths'] 
        obs_mask_paths = None if 'mask_paths' not in observed_data else observed_data['mask_paths']
        observed_data = {k : v.to(device) for k, v in observed_data.items() if isinstance(v, torch.Tensor)}
        for k, v in gt_data.items():
            if isinstance(v, torch.Tensor):
                gt_data[k] = v.to(device)
        cur_batch_size = observed_data[list(observed_data.keys())[0]].size(0)
        T = observed_data[list(observed_data.keys())[0]].size(1)

        if use_overlap_loss and 'seq_interval' not in observed_data:
            print('Must have frame index labels from data to determine overlap')
            exit()

        if cur_batch_size == 3:
            # NOTE: hacky way to avoid bug with pytorch 3x3 matmul problems with batch size 3....
            for k, v in observed_data.items():
                if isinstance(v, torch.Tensor):
                    observed_data[k] = torch.cat([v, v[-1:]], dim=0)
                else:
                    # otherwise it's a list
                    observed_data[k] = v + [v[-1]]
            for k, v in gt_data.items():
                if isinstance(v, torch.Tensor):
                    gt_data[k] = torch.cat([v, v[-1:]], dim=0)
                else:
                    # otherwise it's a list
                    gt_data[k] = v + [v[-1]]
                    if k == 'name':
                        # change name so we don't overwrite
                        gt_data[k][-1] = gt_data[k][-1] + '_copy'

            # obs_img_paths and obs_mask_paths
            if obs_img_paths is not None:
                new_obs_img_paths = []
                for img_paths in obs_img_paths:
                    new_obs_img_paths.append(img_paths + [img_paths[-1]])
                obs_img_paths = new_obs_img_paths
            if obs_mask_paths is not None:
                new_obs_mask_paths = []
                for mask_paths in obs_mask_paths:
                    new_obs_mask_paths.append(mask_paths + [mask_paths[-1]])
                obs_mask_paths = new_obs_mask_paths

            cur_batch_size = 4

        # pass in the last batch index from previous batch is using overlap consistency
        if use_overlap_loss and prev_batch_overlap_res_dict is not None:
            observed_data['prev_batch_overlap_res'] = prev_batch_overlap_res_dict

        seq_names = []
        for gt_idx, gt_name in enumerate(gt_data['name']):
            seq_name = gt_name + '_' + str(int(time.time())) + str(gt_idx)
            Logger.log(seq_name)
            seq_names.append(seq_name)

        cur_z_init_paths = []
        cur_z_final_paths = []
        cur_res_out_paths = []
        for seq_name in seq_names:
            # set current out paths based on sequence name
            if res_out_path is not None:
                cur_res_out_path = os.path.join(res_out_path, seq_name)
                mkdir(cur_res_out_path)
                cur_res_out_paths.append(cur_res_out_path)
        cur_res_out_paths = cur_res_out_paths if len(cur_res_out_paths) > 0 else None
        cur_z_init_paths = cur_z_init_paths if len(cur_z_init_paths) > 0 else None
        cur_z_final_paths = cur_z_final_paths if len(cur_z_final_paths) > 0 else None

        # get body model
        # load in from given path
        Logger.log('Loading SMPL model from %s...' % (args.smpl))
        body_model_path = args.smpl
        fit_gender = body_model_path.split('/')[-2]
        num_betas = 16 if 'betas' not in gt_data else gt_data['betas'].size(2)
        body_model = BodyModel(bm_path=body_model_path,
                                num_betas=num_betas,
                                batch_size=cur_batch_size*T,
                                use_vtx_selector=use_joints2d).to(device)

        if body_model.model_type != 'smplh':
            print('Only SMPL+H model is supported for HuMoR!')
            exit()

        gt_body_paths = None
        if 'gender' in gt_data:
            gt_body_paths = []
            for cur_gender in gt_data['gender']:
                gt_body_path = None
                if args.gt_body_type == 'smplh':
                    gt_body_path = os.path.join(SMPLH_PATH, '%s/model.npz' % (cur_gender))
                gt_body_paths.append(gt_body_path)

        cam_mat = None
        if 'cam_matx' in gt_data:
            cam_mat = gt_data['cam_matx'].to(device)

        #  save meta results information about the optimized bm and GT bm (gender)
        if args.save_results:
            for bidx, cur_res_out_path in enumerate(cur_res_out_paths):
                cur_meta_path = os.path.join(cur_res_out_path, 'meta.txt')
                with open(cur_meta_path, 'w') as f:
                    f.write('optim_bm %s\n' % (body_model_path))
                    if gt_body_paths is None:
                        f.write('gt_bm %s\n' % (body_model_path))
                    else:
                        f.write('gt_bm %s\n' % (gt_body_paths[bidx]))

        # create optimizer
        optimizer = MotionOptimizer(device,
                                    body_model,
                                    num_betas,
                                    cur_batch_size,
                                    T,
                                    [k for k in observed_data.keys()],
                                    all_stage_loss_weights,
                                    pose_prior,
                                    motion_prior,
                                    init_motion_prior,
                                    use_joints2d,
                                    cam_mat,
                                    args.robust_loss,
                                    args.robust_tuning_const,
                                    args.joint2d_sigma,
                                    stage3_tune_init_state=args.stage3_tune_init_state,
                                    stage3_tune_init_num_frames=args.stage3_tune_init_num_frames,
                                    stage3_tune_init_freeze_start=args.stage3_tune_init_freeze_start,
                                    stage3_tune_init_freeze_end=args.stage3_tune_init_freeze_end,
                                    stage3_contact_refine_only=args.stage3_contact_refine_only,
                                    use_chamfer=('points3d' in observed_data),
                                    im_dim=im_dim)

        # run optimizer
        optim_result, per_stage_results = optimizer.run(observed_data,
                                                        data_fps=data_fps,
                                                        lr=args.lr,
                                                        num_iter=args.num_iters,
                                                        lbfgs_max_iter=args.lbfgs_max_iter,
                                                        stages_res_out=cur_res_out_paths,
                                                        fit_gender=fit_gender)

        # save final results
        if cur_res_out_paths is not None:
            save_optim_result(cur_res_out_paths, optim_result, per_stage_results, gt_data, observed_data, args.data_type,
                                optim_floor=optimizer.optim_floor,
                                obs_img_paths=obs_img_paths,
                                obs_mask_paths=obs_mask_paths)

        elapsed_t = time.time() - start_t
        Logger.log('Optimized sequence %d in %f s' % (i, elapsed_t))

        # cache last verts, floor, and betas from last batch index to use in consistency loss
        #   for next batch
        if use_overlap_loss:
            prev_batch_overlap_res_dict = dict()
            prev_batch_overlap_res_dict['verts3d'] = per_stage_results['stage3']['verts3d'][-1].clone().detach()
            prev_batch_overlap_res_dict['betas'] = optim_result['betas'][-1].clone().detach()
            prev_batch_overlap_res_dict['floor_plane'] = optim_result['floor_plane'][-1].clone().detach()
            prev_batch_overlap_res_dict['seq_interval'] = observed_data['seq_interval'][-1].clone().detach()

        if i < (len(data_loader) - 1):
            del optimizer
        del body_model
        del observed_data
        del gt_data
        torch.cuda.empty_cache()

    # if RGB video, stitch together subsequences
    if args.data_type == 'RGB' and args.save_results:
        Logger.log('Saving final results...')
        seq_intervals = dataset.seq_intervals
        save_rgb_stitched_result(seq_intervals, cur_res_out_paths, res_out_path, device,
                                 body_model_path, num_betas, use_joints2d)


if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)