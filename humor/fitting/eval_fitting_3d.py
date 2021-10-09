'''
Viz and evaluation for 3D AMASS tasks.
'''

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import csv, random

import numpy as np

import torch
import torch.nn as nn

from utils.config import SplitLineParser
from utils.logging import mkdir

from fitting.fitting_utils import load_res, prep_res, run_smpl
from fitting.eval_utils import quant_eval_3d, SMPL_SIZES, GRND_PEN_THRESH_LIST, get_grnd_pen_key, AMASS_EVAL_BLACKLIST

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS

from viz.utils import viz_smpl_seq, create_video

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'stage3_results'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results']
OBS_NAME = 'observations'
FPS = 30

# rendering options
IMW, IMH = 1280, 720
CAM_OFFSET = [0.0, 2.25, 0.9]

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    parser.add_argument('--results', type=str, required=True, help='Path to the results directory to run eval on. Should be a directory of directories - one for each sequence.')
    parser.add_argument('--out', type=str, required=True, help='Path to save evaluation results/visualizations to.')

    # quant eval and options
    parser.add_argument('--quant', dest='run_quant_eval', action='store_true', help="If given, runs quantitative evaluation and saves the results.")
    parser.set_defaults(run_quant_eval=False)
    parser.add_argument('--quant-stages', dest='quant_stages', action='store_true', help="If given, runs quantitative evaluation on all stages rather than just final.")
    parser.set_defaults(quant_stages=False)

    # qual eval and options
    parser.add_argument('--qual', dest='run_qual_eval', action='store_true', help="If given, runs qualitative (visualization) evaluation and saves the results. By default this is just the final result and GT.")
    parser.set_defaults(run_qual_eval=False)
    parser.add_argument('--viz-stages', dest='viz_stages', action='store_true', help="If given, visualized intermediate optimization stages.")
    parser.set_defaults(viz_stages=False)
    parser.add_argument('--viz-observation', dest='viz_observation', action='store_true', help="If given, visualizes observations on bodies (e.g. verts).")
    parser.set_defaults(viz_observation=False)
    parser.add_argument('--viz-contacts', dest='viz_contacts', action='store_true', help="Render predicted contacts on the joints")
    parser.set_defaults(viz_contacts=False)

    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles eval ordering")
    parser.set_defaults(shuffle=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args


def main(args):
    print(args)
    mkdir(args.out)

    quant_out_path = qual_out_path = None
    if args.run_quant_eval:
        quant_out_path = os.path.join(args.out, 'eval_quant')
    if args.run_qual_eval:
        qual_out_path = os.path.join(args.out, 'eval_qual')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # collect results directories
    all_result_dirs = [os.path.join(args.results, f) for f in sorted(os.listdir(args.results)) if f[0] != '.']
    all_result_dirs = [f for f in all_result_dirs if os.path.isdir(f)]
    # print(all_result_dirs)
    if args.shuffle:
        random.seed(0)
        random.shuffle(all_result_dirs)

    quant_res_dict = None
    obs_mod = None
    seq_name_list = []
    body_model_dict = dict()
    for ridx, result_dir in enumerate(all_result_dirs):
        seq_name = result_dir.split('/')[-1]
        print('Evaluating %s... %d/%d' % (seq_name, ridx, len(all_result_dirs)))

        seq_name_base = '_'.join(seq_name.split('_')[:-1])
        if seq_name_base in AMASS_EVAL_BLACKLIST:
            print('Skipping %s due to invalid base MVAE results...' % (seq_name))
            continue

        # load in GT        
        gt_res = load_res(result_dir, GT_RES_NAME + '.npz')
        if gt_res is None:
            print('Could not find GT data for %s, skipping...' % (seq_name))
            continue
        T = gt_res['trans'].shape[0]
        num_gt_betas = gt_res['betas'].shape[0]
        gt_res = prep_res(gt_res, device, T)
        # load in pred
        pred_res = load_res(result_dir, PRED_RES_NAME + '.npz')
        if pred_res is None:
            print('Could not find final pred (stage 3) results for %s, skipping...' % (seq_name))
            continue
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

        pred_res = prep_res(pred_res, device, T)

        #
        # create body models for each
        #
        meta_path = os.path.join(result_dir, 'meta.txt')
        if not os.path.exists(meta_path):
            print('Could not find metadata for %s, skipping...' % (seq_name))
            continue
        optim_bm_path = gt_bm_path = None
        with open(meta_path, 'r') as f:
            optim_bm_str = f.readline().strip()
            optim_bm_path = optim_bm_str.split(' ')[1]
            gt_bm_str = f.readline().strip()
            gt_bm_path = gt_bm_str.split(' ')[1]

        gt_bm = pred_bm = None
        if gt_bm_path not in body_model_dict:
            gt_bm = BodyModel(bm_path=gt_bm_path,
                            num_betas=num_gt_betas,
                            batch_size=T).to(device)
            body_model_dict[gt_bm_path] = gt_bm
        if optim_bm_path not in body_model_dict:
            pred_bm = BodyModel(bm_path=optim_bm_path,
                            num_betas=num_pred_betas,
                            batch_size=T).to(device)
            body_model_dict[optim_bm_path] = pred_bm
        gt_bm = body_model_dict[gt_bm_path]
        pred_bm = body_model_dict[optim_bm_path]

        seq_name_list.append(seq_name)

        # also load in results for intermediate stages if needed
        cur_stages_res = None
        if args.quant_stages or args.viz_stages:
            cur_stages_res = dict()
            for stage_name in STAGES_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_res[stage_name] = prep_res(stage_res, device, T)    

        # run through SMPL
        gt_body = run_smpl(gt_res, gt_bm)
        pred_body = run_smpl(pred_res, pred_bm)
        stages_body = None
        if cur_stages_res is not None:
            stages_body = dict()
            for k, v in cur_stages_res.items():
                stages_body[k] = run_smpl(v, pred_bm)

        #
        # Quantitative evaluation
        #
        if args.run_quant_eval:

            obs_path = os.path.join(result_dir, OBS_NAME + '.npz')
            if not os.path.exists:
                print('Could not find observed data for %s, using in quantitative eval...' % (seq_name))
            obs_dict = np.load(obs_path)
            obs_data = dict()
            for obs in obs_dict.files:
                obs_data[obs] = torch.Tensor(obs_dict[obs]).to(device)

            # want to compare gt to pred and stages if desired.
            eval_bodies = [pred_body]
            eval_names = [PRED_RES_NAME]
            eval_res_list = [pred_res]
            if args.quant_stages:
                eval_bodies += [stages_body[k] for k in STAGES_RES_NAMES]
                eval_names += [k for k in STAGES_RES_NAMES]
                eval_res_list += [cur_stages_res[k] for k in STAGES_RES_NAMES]

            if quant_res_dict is None:
                quant_res_dict = dict()
                obs_mod = list(obs_data.keys())[0]
                # initialize result tracking
                for eval_name in eval_names:
                    quant_res_dict[eval_name] = {
                        'joints3d_all' : [], # compare to GT errors
                        'joints3d_ee' : [],
                        'joints3d_legs' : [],
                        'verts3d_all' : [],
                        'mesh3d_all' : [],
                        'contact_acc' : [],
                        'contact_acc_cnt' : [],
                        'accel_mag'  : [],
                        'ground_pen_dist' : []
                    }
                    # add penetration count for all thresholds
                    for pen_thresh in GRND_PEN_THRESH_LIST:
                        cur_pen_key = get_grnd_pen_key(pen_thresh)
                        quant_res_dict[eval_name][cur_pen_key] = []
                        quant_res_dict[eval_name][cur_pen_key + '_cnt'] = []

                    # can also split whatever was observed by visibility
                    quant_res_dict[eval_name][obs_mod + '_vis'] = []
                    quant_res_dict[eval_name][obs_mod + '_occ'] = []
        
            cur_gt_eval = {
                'joints3d' : gt_body.Jtr[:,:len(SMPL_JOINTS)],
                'verts3d' : gt_body.v[:, KEYPT_VERTS],
                'mesh3d' : gt_body.v,
                'contacts' : gt_res['contacts']
            }

            for eval_body, eval_name, eval_res in zip(eval_bodies, eval_names, eval_res_list):
                # evaluate each output result
                cur_pred_eval = {
                    'joints3d' : eval_body.Jtr[:,:len(SMPL_JOINTS)],
                    'verts3d' : eval_body.v[:, KEYPT_VERTS],
                    'mesh3d' : eval_body.v,
                }
                if 'contacts' in pred_res:
                    cur_pred_eval['contacts'] = pred_res['contacts'] # stages don't have contacts so just always use pred
                else:
                    # didn't predict contacts, just use Gt
                    cur_pred_eval['contacts'] = gt_res['contacts']
                quant_eval_3d(quant_res_dict[eval_name], cur_pred_eval, cur_gt_eval, obs_data)
        
        #
        # Qualitative evaluation
        #
        if args.run_qual_eval:
            cur_qual_out_path = os.path.join(qual_out_path, seq_name)
            mkdir(cur_qual_out_path)
            # load in observed data
            obs_data = {'viz_joints' : None, 'viz_points' : None}
            if args.viz_observation:
                obs_path = os.path.join(result_dir, OBS_NAME + '.npz')
                if not os.path.exists:
                    print('Could not find observed data for %s, not visualizing...' % (seq_name))
                else:
                    obs_dict = np.load(obs_path)
                    if 'verts3d' in obs_dict.files:
                        obs_data['viz_points'] = torch.Tensor(obs_dict['verts3d']).to(device)
                    elif 'points3d' in obs_dict.files:
                        obs_data['viz_points'] = torch.Tensor(obs_dict['verts3d']).to(device)
                    if 'joints3d' in obs_dict.files:
                        obs_data['viz_joints'] = torch.Tensor(obs_dict['joints3d']).to(device)

                    if obs_data['viz_joints'] is not None:
                        obs_data['viz_joints'][torch.isinf(obs_data['viz_joints'])] = -1.0 # out of sight
                    if obs_data['viz_points'] is not None:
                        obs_data['viz_points'][torch.isinf(obs_data['viz_points'])] = -1.0 # out of sight

            viz_contacts = pred_res['contacts'] if args.viz_contacts else None
            gt_contacts = gt_res['contacts'] if 'contacts' in gt_res and args.viz_contacts else None

            import trimesh
            cam_pose = trimesh.transformations.rotation_matrix(np.radians(180), (0, 0, 1))
            cam_pose = np.dot(trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0)), cam_pose)
            cam_pose = np.dot(trimesh.transformations.rotation_matrix(np.radians(90), (0, 0, 1)), cam_pose)
            cam_rot = cam_pose[:3,:3]

            have_joints = obs_data['viz_joints'] is not None
            static_out_path  = os.path.join(cur_qual_out_path, 'humor')
            viz_smpl_seq(pred_body,
                        imw=IMH, imh=IMH, fps=30,
                        render_body=True,
                        render_joints=have_joints,
                        render_skeleton=have_joints,
                        render_ground=True,
                        contacts=viz_contacts,
                        joints_seq=obs_data['viz_joints'],
                        points_seq=obs_data['viz_points'],
                        body_alpha=0.5 if have_joints else 1.0,
                        use_offscreen=True,
                        out_path=static_out_path,
                        wireframe=False,
                        RGBA=False,
                        follow_camera=True,
                        cam_offset=CAM_OFFSET,
                        joint_color=[ 0.0, 1.0, 0.0 ],
                        point_color=[0.0, 0.0, 1.0],
                        skel_color=[0.5, 0.5, 0.5],
                        joint_rad=0.02,
                        point_rad=0.02
                    )
            create_video(static_out_path + '/frame_%08d.' + '%s' % ('png'), static_out_path + '.mp4', FPS)

            if args.viz_stages:
                static_out_path  = os.path.join(cur_qual_out_path, 'vposer')
                viz_smpl_seq(stages_body['stage2_results'],
                            imw=IMH, imh=IMH, fps=30,
                            render_body=True,
                            render_joints=have_joints,
                            render_skeleton=have_joints,
                            render_ground=True,
                            contacts=None,
                            joints_seq=obs_data['viz_joints'],
                            points_seq=obs_data['viz_points'],
                            body_alpha=0.5 if have_joints else 1.0,
                            use_offscreen=True,
                            out_path=static_out_path,
                            wireframe=False,
                            RGBA=False,
                            follow_camera=True,
                            cam_offset=CAM_OFFSET,
                            joint_color=[ 0.0, 1.0, 0.0 ],
                            point_color=[0.0, 0.0, 1.0],
                            skel_color=[0.5, 0.5, 0.5],
                            joint_rad=0.02,
                            point_rad=0.02
                        )
                create_video(static_out_path + '/frame_%08d.' + '%s' % ('png'), static_out_path + '.mp4', FPS)

            print(obs_data.keys())
            static_out_path  = os.path.join(cur_qual_out_path, 'gt_obs_only')
            viz_smpl_seq(gt_body,
                        imw=IMH, imh=IMH, fps=30,
                        render_body=True,
                        render_joints=have_joints,
                        render_skeleton=have_joints,
                        render_ground=True,
                        contacts=None,
                        joints_seq=obs_data['viz_joints'],
                        points_seq=obs_data['viz_points'],
                        body_alpha=0.5 if have_joints else 1.0,
                        use_offscreen=True,
                        out_path=static_out_path,
                        wireframe=False,
                        RGBA=False,
                        follow_camera=True,
                        cam_offset=CAM_OFFSET,
                        joint_color=[ 0.0, 1.0, 0.0 ],
                        point_color=[0.0, 0.0, 1.0],
                        skel_color=[0.5, 0.5, 0.5],
                        joint_rad=0.02,
                        point_rad=0.02
                    )
            create_video(static_out_path + '/frame_%08d.' + '%s' % ('png'), static_out_path + '.mp4', FPS)

    if args.run_quant_eval:

        mkdir(quant_out_path)

        # evals that need to be summed/meaned over all entries
        agg_vals = ['joints3d_all', 'joints3d_ee', 'joints3d_legs', 'verts3d_all', 'mesh3d_all']
        agg_vals += [obs_mod + '_vis', obs_mod + '_occ']
        agg_vals += ['accel_mag', 'ground_pen_dist']
        frac_vals = ['contact_acc']
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
                agg_per_seq_means.append(per_seq_means)

                all_val = np.concatenate(eval_res[agg_val], axis=0)
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

            # fraction evals
            for frac_val in frac_vals:
                frac_val_cnt = frac_val + '_cnt'
                per_seq_val = eval_res[frac_val]
                per_seq_cnt = eval_res[frac_val_cnt]
                per_seq_means = [float(seq_val) / seq_cnt for seq_val, seq_cnt in zip(per_seq_val, per_seq_cnt)]
                agg_per_seq_means.append(per_seq_means)

                all_val = np.array(eval_res[frac_val], dtype=float)
                all_cnt = np.array(eval_res[frac_val_cnt], dtype=float)
                all_mean = np.sum(all_val) / np.sum(all_cnt)
                agg_all_means.append(all_mean)
                agg_all_stds.append(-1)
                agg_all_meds.append(-1)
                agg_all_maxs.append(-1)
                agg_all_mins.append(-1)

            # supplemental values
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

            # save
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