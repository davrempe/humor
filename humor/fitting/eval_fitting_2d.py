'''
Evaluation for 2d tasks (iMapper and PROX data)
'''

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time, math, shutil, csv, random

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils.config import SplitLineParser
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues
from utils.torch import load_state
from utils.logging import mkdir

from datasets.prox_dataset import Projection
from fitting.fitting_utils import compute_cam2prior, load_res, prep_res, run_smpl, apply_cam2prior

from fitting.eval_utils import quant_eval_2d
from fitting.eval_utils import IMAP2COMPARE, SMPL2COMPARE
from fitting.eval_utils import IMW, IMH, SMPL_SIZES, GRND_PEN_THRESH_LIST, get_grnd_pen_key
from fitting.eval_utils import RGBD_EVAL_BLACKLIST, RGB_EVAL_BLACKLIST

from body_model.body_model import BodyModel
from body_model.utils import SMPLH_PATH, SMPL_JOINTS

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'stage3_results'
PRED_PRIOR_RES_NAME = 'stage3_results_prior'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results'] # results in camera frame
OBS_NAME = 'observations'
FPS = 30

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    parser.add_argument('--results', type=str, required=True, help='Path to the results directory to run eval on. Should be a directory of directories - one for each sequence.')
    parser.add_argument('--dataset', type=str, required=True, choices=['iMapper', 'PROX', 'PROXD'], help='The dataset we are evaluating results on')
    parser.add_argument('--prox-floors', type=str, default=None, help='Path to the GT floor plane data for PROX. If given, then the GT floor is used for plausibility metrics rather than predicted floor from our method.')
    parser.add_argument('--imapper-floors', type=str, default=None, help='Path to the GT floor plane data for iMapper. If given, then the GT floor is used for plausibility metrics rather than predicted floor from our method.')

    parser.add_argument('--out', type=str, required=True, help='Path to save evaluation results/visualizations to.')

    # quant eval options
    parser.add_argument('--quant-stages', dest='quant_stages', action='store_true', help="If given, runs quantitative evaluation on all stages in addition to the final results.")
    parser.set_defaults(quant_stages=False)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles eval ordering")
    parser.set_defaults(shuffle=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args


def main(args):
    print(args)
    mkdir(args.out)

    quant_out_path = None
    quant_out_path = os.path.join(args.out, 'eval_quant')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # collect our results directories
    all_result_dirs = [os.path.join(args.results, f) for f in sorted(os.listdir(args.results)) if f[0] != '.']
    all_result_dirs = [f for f in all_result_dirs if os.path.isdir(f)]
    if args.shuffle:
        random.seed(0)
        random.shuffle(all_result_dirs)

    quant_res_dict = None
    seq_name_list = []
    body_model_dict = dict()
    for residx, result_dir in enumerate(all_result_dirs):
        seq_name = '_'.join(result_dir.split('/')[-1].split('_')[:-1])
        print('Evaluating %s %d / %d...' % (seq_name, residx, len(all_result_dirs)))
        if args.dataset == 'PROXD' and seq_name in RGBD_EVAL_BLACKLIST:
            print('Skipping %s due to invalid PROXD results...' % (seq_name))
            continue
        if args.dataset in ['iMapper', 'PROX'] and seq_name in RGB_EVAL_BLACKLIST:
            print('Skipping %s due to invalid VIBE results...' % (seq_name))
            continue
        scene_name = '_'.join(seq_name.split('_')[:-1]) # the full sequence before splitting up
        print(scene_name)

        obs_dict = load_res(result_dir, OBS_NAME + '.npz')
        cur_img_paths = obs_dict['img_paths'] # used to load in results from baselines
        cur_mask_paths = obs_dict['mask_paths'] # used in visualizaiton
        cur_frame_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in cur_img_paths]

        # load in ground truth info
        gt_res = load_res(result_dir, GT_RES_NAME + '.npz')
        if gt_res is None:
            print('Could not find GT data for %s, skipping...' % (seq_name))
            continue

        # get camera intrinsics
        cam_fx = gt_res['cam_mtx'][0, 0]
        cam_fy = gt_res['cam_mtx'][1, 1]
        cam_cx = gt_res['cam_mtx'][0, 2]
        cam_cy = gt_res['cam_mtx'][1, 2]
        cam_intrins = (cam_fx, cam_fy, cam_cx, cam_cy)

        # load in GT occlusions for imapper data
        if args.dataset == 'iMapper':
            gt_joints = gt_res['joints3d']
            # also map joints (AND occlusion) to the comparison skeleton
            gt_joints = gt_joints[:, IMAP2COMPARE, :]
            gt_res['joints3d_comp'] = torch.Tensor(gt_joints).to(device)

        gt_floor_plane = None
        if args.dataset in ['PROX', 'PROXD'] and args.prox_floors is not None:
            cur_prox_scene_name = scene_name.split('_')[0]
            floor_path = os.path.join(args.prox_floors, cur_prox_scene_name + '.txt')
            if not os.path.exists(floor_path):
                print('Could not find floor plane for scene at %s...using predicted' % (floor_path))
            else:
                print('Using GT floors from PROX dataset for plausibility metrics rather than predicted!')
                with open(floor_path, 'r') as floor_f:
                    floor_str = floor_f.readline()
                floor_arr = [float(x) for x in floor_str.split(' ')]
                gt_floor_plane = torch.Tensor(floor_arr).to(device)
        elif args.dataset == 'iMapper' and args.imapper_floors is not None:
            cur_imapper_scene_name = scene_name.split('_')[0]
            floor_path = os.path.join(args.imapper_floors, cur_imapper_scene_name + '.txt')
            if not os.path.exists(floor_path):
                print('Could not find floor plane for scene at %s...using predicted' % (floor_path))
            else:
                print('Using GT floors from iMapper dataset for plausibility metrics rather than predicted!')
                with open(floor_path, 'r') as floor_f:
                    floor_str = floor_f.readline()
                floor_arr = [float(x) for x in floor_str.split(' ')]
                gt_floor_plane = torch.Tensor(floor_arr).to(device)

        # load in our prediction
        pred_res = load_res(result_dir, PRED_RES_NAME + '.npz')
        if pred_res is None:
            print('Could not find final pred (stage 3) results for %s, skipping...' % (seq_name))
            continue
        T = pred_res['trans'].shape[0]
        num_pred_betas = pred_res['betas'].shape[0]
        # check if have any nans valid
        for smpk in SMPL_SIZES.keys():
            cur_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res[smpk])))).item() == 0)
            if not cur_valid:
                print('Found NaNs in prediction for %s, filling with zeros...' % (smpk))
                # print(pred_res[smpk].shape)
                if smpk == 'betas':
                    pred_res[smpk] = np.zeros((pred_res[smpk].shape[0]), dtype=float)
                else:
                    pred_res[smpk] = np.zeros((T, pred_res[smpk].shape[1]), dtype=float)
        floor_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res['floor_plane'])))).item() == 0)
        if not floor_valid:
            print('Predicted floor is NaN, replacing with up.')
            pred_res['floor_plane'] = np.array([0.0, -1.0, 0.0, 0.0])

        pred_res = prep_res(pred_res, device, T)
        pred_floor_plane = torch.Tensor(pred_res['floor_plane']).to(device)

        # load stages results if needed 
        cur_stages_res = None
        if args.quant_stages:
            cur_stages_res = dict()
            for stage_name in STAGES_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_res[stage_name] = prep_res(stage_res, device, T)

        #
        # create body models for each
        #
        meta_path = os.path.join(result_dir, 'meta.txt')
        if not os.path.exists(meta_path):
            print('Could not find metadata for %s, skipping...' % (seq_name))
            continue
        optim_bm_path = None
        with open(meta_path, 'r') as f:
            optim_bm_str = f.readline().strip()
            optim_bm_path = optim_bm_str.split(' ')[1]

        # our model
        pred_bm = None
        if optim_bm_path not in body_model_dict:
            pred_bm = BodyModel(bm_path=optim_bm_path,
                            num_betas=num_pred_betas,
                            batch_size=T).to(device)
            body_model_dict[optim_bm_path] = pred_bm
        pred_bm = body_model_dict[optim_bm_path]

        # we are using this sequence for sure
        seq_name_list.append(seq_name)

        # run through SMPL
        pred_body = run_smpl(pred_res, pred_bm)
        # get body smpl joints
        pred_body_joints = pred_body.Jtr[:, :len(SMPL_JOINTS)]
        pred_res['joints3d_smpl'] = pred_body_joints
        if args.dataset == 'iMapper':
            # get comparison 3d joints
            pred_comp_joints = pred_body_joints[:, SMPL2COMPARE, :]
            pred_res['joints3d_comp'] = pred_comp_joints
        
        stages_body = None
        if cur_stages_res is not None:
            stages_body = dict()
            for k, v in cur_stages_res.items():
                stages_body[k] = run_smpl(v, pred_bm)
                # get body smpl joints
                stage_body_joints = stages_body[k].Jtr[:, :len(SMPL_JOINTS)]
                cur_stages_res[k]['joints3d_smpl'] = stage_body_joints
                if args.dataset == 'iMapper':
                    # get comparison 3d joints
                    stage_comp_joints = stage_body_joints[:, SMPL2COMPARE, :]
                    cur_stages_res[k]['joints3d_comp'] = stage_comp_joints

        #
        # Load in masks
        # 
        mask_arr = np.zeros((T, IMH, IMW), dtype=np.float32)
        for midx, mask_path in enumerate(cur_mask_paths): 
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            if args.dataset in ['PROX', 'PROXD']:
                # need to flip to make comparable
                mask = cv2.flip(mask, 1)
            mask_arr[midx] = mask

        #
        # Quantitative evaluation
        #
        #

        # collect all pred and stages and all go through the same quant eval function
        # always evaluate our method predictions
        eval_res_list = [pred_res]
        eval_name_list = [PRED_RES_NAME]
        # if desired, evaluate stages
        if args.quant_stages:
            eval_res_list += [cur_stages_res[k] for k in STAGES_RES_NAMES]
            eval_name_list += [k for k in STAGES_RES_NAMES]

        if quant_res_dict is None:
            quant_res_dict = dict()
            # initialize result trakcing
            for eval_name in eval_name_list:
                quant_res_dict[eval_name] = {
                    'accel_mag'  : [],
                    'accel_mag_align' : [],
                    'ground_pen_dist' : []
                }
                # add penetration count for all thresholds
                for pen_thresh in GRND_PEN_THRESH_LIST:
                    cur_pen_key = get_grnd_pen_key(pen_thresh)
                    quant_res_dict[eval_name][cur_pen_key] = []
                    quant_res_dict[eval_name][cur_pen_key + '_cnt'] = []
                if args.dataset == 'iMapper':
                    quant_res_dict[eval_name]['joints3d_all'] = [] # must have GT joints
                    quant_res_dict[eval_name]['joints3d_ee'] = []
                    quant_res_dict[eval_name]['joints3d_legs'] = []
                    quant_res_dict[eval_name]['joints3d_vis'] = []
                    quant_res_dict[eval_name]['joints3d_occ'] = []
                    quant_res_dict[eval_name]['joints3d_align_all'] = []
                    quant_res_dict[eval_name]['joints3d_align_ee'] = []
                    quant_res_dict[eval_name]['joints3d_align_legs'] = []
                    quant_res_dict[eval_name]['joints3d_align_vis'] = []
                    quant_res_dict[eval_name]['joints3d_align_occ'] = []

        eval_floor_plane = gt_floor_plane if gt_floor_plane is not None else pred_floor_plane
        for eval_res, eval_name in zip(eval_res_list, eval_name_list):
            pred_joints_smpl = eval_res['joints3d_smpl']
            pred_joints_comp = gt_joints_comp = vis_mask = None
            if args.dataset == 'iMapper':
                pred_joints_comp = eval_res['joints3d_comp']
                gt_joints_comp = gt_res['joints3d_comp']
                vis_mask = mask_arr
            quant_eval_2d(quant_res_dict[eval_name], pred_joints_smpl, eval_floor_plane,
                                    pred_joints_comp=pred_joints_comp,
                                    gt_joints_comp=gt_joints_comp,
                                    vis_mask=vis_mask,
                                    cam_intrins=cam_intrins)
        
    mkdir(quant_out_path)

    # evals that need to be summed/meaned over all entries
    agg_vals = ['accel_mag', 'accel_mag_align', 'ground_pen_dist']
    if args.dataset == 'iMapper':
        agg_vals += ['joints3d_all', 'joints3d_ee', 'joints3d_legs', 'joints3d_vis', 'joints3d_occ',
                        'joints3d_align_all', 'joints3d_align_ee', 'joints3d_align_legs', 'joints3d_align_vis', 'joints3d_align_occ']
    frac_vals = []
    for pen_thresh in GRND_PEN_THRESH_LIST:
        frac_vals.append(get_grnd_pen_key(pen_thresh))

    supp_vals = ['ground_pen_dist_normalized', 'ground_pen_mean_agg_frac'] # values computed after from everything else

    per_seq_out_vals = agg_vals + frac_vals
    out_vals = agg_vals + frac_vals + supp_vals

    eval_names_out = []
    eval_means_out = []
    eval_maxs_out = []
    eval_meds_out = []
    for eval_name, eval_res in quant_res_dict.items():
        eval_names_out.append(eval_name)

        # agg evals
        agg_per_seq_means = []
        agg_all_means = []
        agg_all_stds = []
        agg_all_meds = []
        agg_all_maxs = []
        agg_all_mins = []
        for agg_val in agg_vals:
            per_seq_val = eval_res[agg_val]
            per_seq_means = []
            for x in per_seq_val:
                seq_mean = np.mean(x) if x.shape[0] > 0 else -1
                per_seq_means.append(seq_mean)
            # print(per_seq_means)
            agg_per_seq_means.append(per_seq_means)

            all_val = np.concatenate(eval_res[agg_val], axis=0)
            # print(all_val.shape)
            if all_val.shape[0] == 0:
                dummy_res = -1
                agg_all_means.append(dummy_res)
                agg_all_stds.append(dummy_res)
                agg_all_meds.append(dummy_res)
                agg_all_maxs.append(dummy_res)
                agg_all_mins.append(dummy_res)
            else:
                all_mean = np.mean(all_val)
                agg_all_means.append(all_mean)
                all_std = np.std(all_val)
                agg_all_stds.append(all_std)
                all_median = np.median(all_val)
                agg_all_meds.append(all_median)
                all_max = np.amax(all_val)
                agg_all_maxs.append(all_max)
                all_min = np.amin(all_val)
                agg_all_mins.append(all_min)

            # print(all_mean)

        # fraction evals
        for frac_val in frac_vals:
            frac_val_cnt = frac_val + '_cnt'
            per_seq_val = eval_res[frac_val]
            per_seq_cnt = eval_res[frac_val_cnt]
            per_seq_means = [float(seq_val) / seq_cnt for seq_val, seq_cnt in zip(per_seq_val, per_seq_cnt)]
            # print(per_seq_means)
            agg_per_seq_means.append(per_seq_means)

            all_val = np.array(eval_res[frac_val], dtype=float)
            all_cnt = np.array(eval_res[frac_val_cnt], dtype=float)
            all_mean = np.sum(all_val) / np.sum(all_cnt)
            agg_all_means.append(all_mean)
            agg_all_stds.append(0.0)
            agg_all_meds.append(0.0)
            agg_all_maxs.append(0.0)
            agg_all_mins.append(0.0)

        # supplemental values
        # 'ground_pen_dist_normalized', 'ground_pen_mean_agg_frac'
        ground_pen_dist_normalized_mean = agg_all_means[out_vals.index('ground_pen_dist')]*agg_all_means[out_vals.index(get_grnd_pen_key(0.0))]
        agg_all_means.append(ground_pen_dist_normalized_mean)
        ground_pen_dist_normalized_med = agg_all_meds[out_vals.index('ground_pen_dist')]*agg_all_means[out_vals.index(get_grnd_pen_key(0.0))]
        agg_all_meds.append(ground_pen_dist_normalized_med)
        ground_pen_frac_sum = 0.0
        for pen_thresh in GRND_PEN_THRESH_LIST:
            ground_pen_frac_sum += agg_all_means[out_vals.index(get_grnd_pen_key(pen_thresh))]
        ground_pen_mean_agg_frac = ground_pen_frac_sum / len(GRND_PEN_THRESH_LIST)
        agg_all_means.append(ground_pen_mean_agg_frac)
        agg_all_meds.append(-1)

        agg_all_stds.extend([-1, -1])
        agg_all_maxs.extend([-1, -1])
        agg_all_mins.extend([-1, -1])

        eval_means_out.append(agg_all_means)
        eval_maxs_out.append(agg_all_maxs)
        eval_meds_out.append(agg_all_meds)

        stage_out_path = os.path.join(quant_out_path, eval_name + '_per_seq_mean.csv')
        with open(stage_out_path, 'w') as f:
            csvw = csv.writer(f, delimiter=',')
            # write heading
            csvw.writerow(['seq_name'] + per_seq_out_vals)
            # write data
            for j, seq_name in enumerate(seq_name_list):
                cur_row = [agg_per_seq_means[vidx][j] for vidx in range(len(per_seq_out_vals))] 
                csvw.writerow([seq_name] + cur_row)

        stats_name_list = ['mean', 'std', 'median', 'max', 'min']
        stats_list = [agg_all_means, agg_all_stds, agg_all_meds, agg_all_maxs, agg_all_mins]
        for stat_name, stat_data in zip(stats_name_list, stats_list):
            agg_out_path = os.path.join(quant_out_path, eval_name + '_agg_%s.csv' % (stat_name))
            with open(agg_out_path, 'w') as f:
                csvw = csv.writer(f, delimiter=',')
                # write heading
                csvw.writerow(out_vals)
                # write data
                csvw.writerow(stat_data)

    #  one file that saves all means together for easy compare
    stats_name_list = ['mean', 'max', 'median']
    stats_list = [eval_means_out, eval_maxs_out, eval_meds_out]
    for stat_name, stat_data in zip(stats_name_list, stats_list):
        compare_out_path = os.path.join(quant_out_path, 'compare_%s.csv' % (stat_name))
        with open(compare_out_path, 'w') as f:
            csvw = csv.writer(f, delimiter=',')
            # write heading
            csvw.writerow(['method'] + out_vals)
            # write data
            for eval_name_out, eval_stat_out in zip(eval_names_out, stat_data):
                csvw.writerow([eval_name_out] + eval_stat_out)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)