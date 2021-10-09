'''
Visualization for RGB results.
'''

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time, math, shutil, csv, random

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.config import SplitLineParser
from utils.transforms import rotation_matrix_to_angle_axis, batch_rodrigues
from utils.torch import load_state
from utils.logging import mkdir

from fitting.fitting_utils import load_res, prep_res, run_smpl
from fitting.eval_utils import SMPL_SIZES

from body_model.body_model import BodyModel
from body_model.utils import SMPL_PATH, SMPLH_PATH, SMPL_JOINTS, SMPLX_PATH

from viz.utils import viz_smpl_seq, viz_results, create_gif, create_video, create_multi_comparison_images, smpl_connections, imapper_connections, comp_connections
from viz.mesh_viewer import COMPRESS_PARAMS

J_BODY = len(SMPL_JOINTS)-1 # no root

GT_RES_NAME = 'gt_results'
PRED_RES_NAME = 'stage3_results'
PRED_PRIOR_RES_NAME = 'stage3_results_prior'
STAGES_RES_NAMES = ['stage1_results', 'stage2_results', 'stage3_init_results'] # results in camera frame
STAGES_PRIOR_RES_NAMES = ['stage2_results_prior', 'stage3_init_results_prior'] # results in prior frame (w.r.t final floor fit)
FINAL_RES_NAME = 'final_results'
FINAL_PRIOR_RES_NAME = 'final_results_prior'
OBS_NAME = 'observations'
FPS = 30

# visualization options
GROUND_ALPHA = 1.0
BODY_ALPHA = None # use to make body mesh translucent
IM_EXTN = 'jpg' # png # to use for rendering jpg saves a lot of space

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    parser.add_argument('--results', type=str, required=True, help='Path to the results_out directory from fitting to run viz on.')
    parser.add_argument('--out', type=str, required=True, help='Path to save visualizations to.')

    # visualization options
    parser.add_argument('--viz-final-only', dest='viz_final_only', action='store_true', help="If given only visualize the final full sequence result and not the subsequences.")
    parser.set_defaults(viz_final_only=False)
    parser.add_argument('--viz-stages', dest='viz_stages', action='store_true', help="If given, visualizes intermediate optimization stages and comparison to final pred.")
    parser.set_defaults(viz_stages=False)
    parser.add_argument('--viz-prior-frame', dest='viz_prior_frame', action='store_true', help="If given, also visualizes results in the HuMoR canonical coordinate frame.")
    parser.set_defaults(viz_prior_frame=False)
    parser.add_argument('--viz-obs-2d', dest='viz_obs_2d', action='store_true', help="If given, visualizes 2D joint observations on top of og video")
    parser.set_defaults(viz_obs_2d=False)
    parser.add_argument('--viz-no-render-cam-body', dest='viz_render_cam_body', action='store_false', help="If given, does not render body mesh from camera view")
    parser.set_defaults(viz_render_cam_body=True)
    parser.add_argument('--viz-pred-floor', dest='viz_pred_floor', action='store_true', help="Render the predicted floor from the camera view.")
    parser.set_defaults(viz_pred_floor=False)
    parser.add_argument('--viz-contacts', dest='viz_contacts', action='store_true', help="Render predicted contacts on the joints")
    parser.set_defaults(viz_contacts=False)
    parser.add_argument('--viz-wireframe', dest='viz_wireframe', action='store_true', help="Render body and floor in wireframe")
    parser.set_defaults(viz_wireframe=False)
    parser.add_argument('--viz-bodies-static', type=int, default=None, help="If given, renders all body predictions at once at this given frame interval interval.")
    parser.add_argument('--viz-no-bg', dest='viz_bg', action='store_false', help="If given will not overlay the rendering on top of OG video.")
    parser.set_defaults(viz_bg=True)

    parser.add_argument('--viz-render-width', type=int, default=1280, help="Width of rendered output images")
    parser.add_argument('--viz-render-height', type=int, default=720, help="Width of rendered output images")

    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles viz ordering")
    parser.set_defaults(shuffle=False)

    parser.add_argument('--flip-img', dest='flip_img', action='store_true', help="Flips the loaded image about y-axis. This is useful for PROX result.")
    parser.set_defaults(flip_img=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args


def main(args):
    print(args)
    mkdir(args.out)
    qual_out_path = args.out
    D_IMW, D_IMH = args.viz_render_width, args.viz_render_height

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # collect our results directories
    all_result_dirs = [os.path.join(args.results, f) for f in sorted(os.listdir(args.results)) if f[0] != '.']
    all_result_dirs = [f for f in all_result_dirs if os.path.isdir(f)]
    if args.shuffle:
        random.seed(0)
        random.shuffle(all_result_dirs)
    print(all_result_dirs)

    seq_name_list = []
    body_model_dict = dict()
    for residx, result_dir in enumerate(all_result_dirs):
        seq_name = result_dir.split('/')[-1]
        is_final_res = seq_name == 'final_results'
        if not is_final_res:
            if args.viz_final_only:
                continue
            seq_name = '_'.join(result_dir.split('/')[-1].split('_')[:-1])
        print('Visualizing %s %d / %d...' % (seq_name, residx, len(all_result_dirs)))

        obs_dict = load_res(result_dir, OBS_NAME + '.npz')
        cur_img_paths = obs_dict['img_paths'] # used to load in results from baselines
        cur_frame_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in cur_img_paths]

        # load in humor prediction
        pred_res = load_res(result_dir, PRED_RES_NAME + '.npz')
        if pred_res is None:
            print('Could not find final pred (stage 3) results for %s, skipping...' % (seq_name))
            continue
        T = pred_res['trans'].shape[0]
        # check if have any nans valid
        for smpk in SMPL_SIZES.keys():
            cur_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res[smpk])))).item() == 0)
            if not cur_valid:
                print('Found NaNs in prediction for %s, filling with zeros...' % (smpk))
                # print(pred_res[smpk].shape)
                if smpk == 'betas':
                    pred_res[smpk] = np.zeros((pred_res[smpk].shape[0]), dtype=np.float)
                else:
                    pred_res[smpk] = np.zeros((T, pred_res[smpk].shape[1]), dtype=np.float)
        floor_valid = (torch.sum(torch.logical_not(torch.isfinite(torch.Tensor(pred_res['floor_plane'])))).item() == 0)
        if not floor_valid:
            print('Predicted floor is NaN, replacing with up.')
            pred_res['floor_plane'] = np.array([0.0, -1.0, 0.0, 0.0])

        pred_res = prep_res(pred_res, device, T)
        num_pred_betas = pred_res['betas'].size(1)

        pred_floor_plane = torch.Tensor(pred_res['floor_plane']).to(device)

        # humor prediction in prior frame
        pred_res_prior = None
        if args.viz_prior_frame:
            pred_res_prior = load_res(result_dir, PRED_PRIOR_RES_NAME + '.npz')
            if pred_res_prior is None:
                    print('Could not find final prior pred (stage 3) results for %s, skipping...' % (seq_name))
                    continue
            pred_res_prior = prep_res(pred_res_prior, device, T)

        # load stages results if needed 
        cur_viz_stages = args.viz_stages and not is_final_res
        cur_stages_res = None
        if cur_viz_stages:
            cur_stages_res = dict()
            for stage_name in STAGES_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_res[stage_name] = prep_res(stage_res, device, T)

        # load prior stages results if needed
        cur_stages_prior_res = None
        if args.viz_prior_frame and cur_viz_stages:
            cur_stages_prior_res = dict()
            for stage_name in STAGES_PRIOR_RES_NAMES:
                stage_res = load_res(result_dir, stage_name + '.npz')
                if stage_res is None:
                    print('Could not find results for stage %s of %s, skipping...' % (stage_name, seq_name))
                    continue
                cur_stages_prior_res[stage_name] = prep_res(stage_res, device, T)   

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

        # humor model
        pred_bm = None
        if optim_bm_path not in body_model_dict:
            pred_bm = BodyModel(bm_path=optim_bm_path,
                            num_betas=num_pred_betas,
                            batch_size=T).to(device)
            if not is_final_res:
                # final results will be different length, so want to re-load for subsequences
                body_model_dict[optim_bm_path] = pred_bm
        if not is_final_res:
            pred_bm = body_model_dict[optim_bm_path]

        # we are using this sequence for sure
        seq_name_list.append(seq_name)

        # run through SMPL
        pred_body = run_smpl(pred_res, pred_bm)
        
        stages_body = None
        if cur_stages_res is not None:
            stages_body = dict()
            for k, v in cur_stages_res.items():
                stages_body[k] = run_smpl(v, pred_bm)
                # get body smpl joints
                stage_body_joints = stages_body[k].Jtr[:, :len(SMPL_JOINTS)]
                cur_stages_res[k]['joints3d_smpl'] = stage_body_joints

        # prior frame through SMPL
        pred_prior_body = None
        if pred_res_prior is not None:
            pred_prior_body = run_smpl(pred_res_prior, pred_bm)

        stages_prior_body = None
        if cur_stages_prior_res is not None:
            stages_prior_body = dict()
            for k, v in cur_stages_prior_res.items():
                stages_prior_body[k] = run_smpl(v, pred_bm)

        # load in image frames
        IMW, IMH = None, None
        img_arr = np.zeros((T, D_IMH, D_IMW, 3), dtype=np.float32)
        for imidx, img_path in enumerate(cur_img_paths):
            img = cv2.imread(img_path)
            if args.flip_img:
                img = cv2.flip(img, 1)
            IMH, IMW, _ = img.shape
            img = cv2.resize(img, (D_IMW, D_IMH), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32)[:, :, ::-1] / 255.0
            img_arr[imidx] = img

        # load in camera info
        gt_res = None
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
        # print(cam_intrins)
        x_frac = float(D_IMW) / IMW
        y_frac = float(D_IMH) / IMH
        cam_intrins_down = (cam_fx*x_frac, cam_fy*y_frac, cam_cx*x_frac, cam_cy*y_frac)
        
        #
        # Qualitative evaluation
        #
        cur_qual_out_path = os.path.join(qual_out_path, seq_name)
        mkdir(cur_qual_out_path)

        # always use final fit ground plane for visualization
        viz_ground_plane = None
        if args.viz_pred_floor:
            viz_ground_plane = pred_res['floor_plane'] if args.viz_pred_floor else None
        render_ground_plane = viz_ground_plane is not None

        viz_points = None
        viz_point_color =[0.0, 1.0, 0.0]
        if args.viz_obs_2d:
            viz_joints2d = obs_dict['joints2d']
            viz_joints2d[:,:,0] = viz_joints2d[:,:,0]*x_frac
            viz_joints2d[:,:,1] = viz_joints2d[:,:,1]*y_frac
        # use contacts if desired
        viz_contacts = pred_res['contacts'] if args.viz_contacts else None

        # always render OG video
        og_out_path = os.path.join(cur_qual_out_path, 'og_video')
        mkdir(og_out_path)
        for fidx, img in enumerate(img_arr):
            img = (img*255.0).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(og_out_path, 'frame_%08d.%s' % (fidx, IM_EXTN)), img_bgr, COMPRESS_PARAMS)
        create_video(og_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), og_out_path + '.mp4', FPS)

        mask_viz = None
        scene_viz = None

        img_viz = img_arr if args.viz_bg else None

        if args.viz_obs_2d:
            # visualize Openpose (with really high confidence) on top of video
            og_2d_obs_out_path = os.path.join(cur_qual_out_path, 'og_video_2d_obs')
            mkdir(og_2d_obs_out_path)
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(float(D_IMW)*0.01, float(D_IMH)*0.01), dpi=100, frameon=False)
            for fidx, img in enumerate(img_arr):
                plt.imshow(img, aspect='auto')
                valid_mask = viz_joints2d[fidx, :, 2] > 0.7
                plt.scatter(viz_joints2d[fidx, valid_mask, 0], viz_joints2d[fidx, valid_mask, 1], c='lime', s=100)
                ax = plt.gca()
                plt.axis('off')
                cur_joint2d_out = os.path.join(og_2d_obs_out_path, 'frame_%08d.png' % (fidx))
                plt.savefig(cur_joint2d_out, bbox_inches='tight', pad_inches=0)
                plt.clf()
            plt.close(fig)
        
            create_video(og_2d_obs_out_path + '/frame_%08d.' + '%s' % ('png'), og_2d_obs_out_path + '.mp4', FPS)
        
        # render camera-view prediction
        pred_out_path = os.path.join(cur_qual_out_path, 'final_pred')
        viz_smpl_seq(pred_body, imw=D_IMW, imh=D_IMH, fps=FPS,
                        render_body=args.viz_render_cam_body,
                        render_bodies_static=args.viz_bodies_static,
                        render_points_static=args.viz_bodies_static,
                        render_joints=(args.viz_wireframe or BODY_ALPHA is not None),
                        render_skeleton=BODY_ALPHA is not None,
                        skel_color=[0.5, 0.5, 0.5],
                        joint_rad=0.02,
                        render_ground=render_ground_plane,
                        ground_plane=viz_ground_plane,
                        ground_alpha=GROUND_ALPHA,
                        body_alpha=BODY_ALPHA,
                        static_meshes=scene_viz,
                        points_seq=viz_points,
                        point_color=viz_point_color,
                        contacts=viz_contacts,
                        use_offscreen=True,
                        out_path=pred_out_path,
                        wireframe=args.viz_wireframe,
                        RGBA=True,
                        point_rad=0.004,
                        follow_camera=False,
                        camera_intrinsics=cam_intrins_down,
                        img_seq=img_viz,
                        mask_seq=mask_viz,
                        img_extn=IM_EXTN)
        create_video(pred_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), pred_out_path + '.mp4', FPS)

        # always comparison of prediction and OG video
        og_pred_comp_path = os.path.join(cur_qual_out_path, 'comp_og_final_pred')
        create_multi_comparison_images([og_out_path, pred_out_path], 
                                og_pred_comp_path,
                                ['Input', 'Final'],
                                extn=IM_EXTN)
        create_video(og_pred_comp_path + '/frame_%08d.' + '%s' % (IM_EXTN), og_pred_comp_path + '.mp4', FPS)

        # render all stages then comparison of og vid, stage2 and final
        if cur_viz_stages:
            for k, stage_body in stages_body.items():
                if not cur_viz_stages and k != STAGES_RES_NAMES[1]:
                    continue # only want stage 2 in this case
                stage_out_path = os.path.join(cur_qual_out_path, k)
                viz_smpl_seq(stage_body, imw=D_IMW, imh=D_IMH, fps=FPS,
                            render_body=True,
                            render_bodies_static=args.viz_bodies_static,
                            render_joints=args.viz_wireframe,
                            render_skeleton=False,
                            render_ground=render_ground_plane,
                            ground_plane=viz_ground_plane,
                            ground_alpha=GROUND_ALPHA,
                            body_alpha=BODY_ALPHA,
                            static_meshes=scene_viz,
                            points_seq=viz_points,
                            point_color=viz_point_color,
                            use_offscreen=True,
                            out_path=stage_out_path,
                            wireframe=args.viz_wireframe,
                            RGBA=True,
                            point_rad=0.004,
                            follow_camera=False,
                            camera_intrinsics=cam_intrins_down,
                            img_seq=img_viz,
                            mask_seq=mask_viz,
                            img_extn=IM_EXTN)
                create_video(stage_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), stage_out_path + '.mp4', FPS)

            # create comparison
            stage2_out_path = os.path.join(cur_qual_out_path, STAGES_RES_NAMES[1])
            if cur_viz_stages:
                pred_stage_comp_path = os.path.join(cur_qual_out_path, 'comp_final_pred_stages')
                create_multi_comparison_images([og_out_path, stage2_out_path, pred_out_path],
                                    pred_stage_comp_path,
                                    ['Input', 'Stage2', 'Final'],
                                    extn=IM_EXTN)
                create_video(pred_stage_comp_path + '/frame_%08d.' + '%s' % (IM_EXTN), pred_stage_comp_path + '.mp4', FPS)

        del img_viz
        del img_arr

        # repeat in prior frame for everything
        if args.viz_prior_frame:
            cam_rot = None
            prior_cam_offset = [0.0, 2.2, 0.9]
            prior_frame_use_follow = True
            # final prediction
            pred_prior_out_path = os.path.join(cur_qual_out_path, 'final_pred_prior')
            viz_smpl_seq(pred_prior_body, imw=D_IMH, imh=D_IMH, fps=FPS,
                            render_body=True,
                            render_bodies_static=args.viz_bodies_static,
                            render_joints=(args.viz_wireframe or BODY_ALPHA is not None),
                            render_skeleton=BODY_ALPHA is not None,
                            body_alpha=BODY_ALPHA,
                            skel_color=[0.5, 0.5, 0.5],
                            joint_rad=0.02,
                            render_ground=True,
                            contacts=viz_contacts,
                            use_offscreen=True,
                            out_path=pred_prior_out_path,
                            wireframe=args.viz_wireframe,
                            RGBA=True,
                            follow_camera=prior_frame_use_follow,
                            cam_offset=prior_cam_offset,
                            cam_rot=cam_rot,
                            img_extn=IM_EXTN)
            create_video(pred_prior_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), pred_prior_out_path + '.mp4', FPS)

            # always comparison of prediction and OG video
            og_pred_prior_comp_path = os.path.join(cur_qual_out_path, 'comp_og_final_pred_prior')
            create_multi_comparison_images([og_out_path, pred_out_path, pred_prior_out_path], 
                                og_pred_prior_comp_path,
                                ['Input', 'FinalCam', 'FinalPrior'],
                                extn=IM_EXTN)
            create_video(og_pred_prior_comp_path + '/frame_%08d.' + '%s' % (IM_EXTN), og_pred_prior_comp_path + '.mp4', FPS)

            # render all stages then comparison of og vid, stage2 and final
            if cur_viz_stages:
                for k, stage_prior_body in stages_prior_body.items():
                    if not cur_viz_stages and k != STAGES_PRIOR_RES_NAMES[0]:
                        continue # only want stage 2 in this case
                    stage_prior_out_path = os.path.join(cur_qual_out_path, k)
                    viz_smpl_seq(stage_prior_body, imw=D_IMH, imh=D_IMH, fps=FPS,
                                render_body=True,
                                render_bodies_static=args.viz_bodies_static,
                                render_joints=args.viz_wireframe,
                                render_skeleton=False,
                                render_ground=True,
                                use_offscreen=True,
                                out_path=stage_prior_out_path,
                                wireframe=args.viz_wireframe,
                                RGBA=True,
                                follow_camera=prior_frame_use_follow,
                                cam_offset=prior_cam_offset,
                                cam_rot=cam_rot,
                                img_extn=IM_EXTN)
                    create_video(stage_prior_out_path + '/frame_%08d.' + '%s' % (IM_EXTN), stage_prior_out_path + '.mp4', FPS)

                # create comparison
                stage2_prior_out_path = os.path.join(cur_qual_out_path, STAGES_PRIOR_RES_NAMES[0])
                if cur_viz_stages:
                    pred_stage_prior_comp_path = os.path.join(cur_qual_out_path, 'comp_final_pred_prior_stages')
                    create_multi_comparison_images([og_out_path, stage2_prior_out_path, pred_prior_out_path],
                                        pred_stage_prior_comp_path,
                                        ['Input', 'Stage2Prior', 'FinalPrior'],
                                        extn=IM_EXTN)
                    create_video(pred_stage_prior_comp_path + '/frame_%08d.' + '%s' % (IM_EXTN), pred_stage_prior_comp_path + '.mp4', FPS)

        del pred_bm

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)