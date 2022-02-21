
import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import importlib, time

import numpy as np

import torch
from torch.utils.data import DataLoader

from utils.config import TestConfig
from utils.logging import Logger, class_name_to_file_name, mkdir, cp_files
from utils.torch import get_device, save_state, load_state
from utils.stats import StatTracker
from utils.transforms import rotation_matrix_to_angle_axis
from body_model.utils import SMPL_JOINTS
from datasets.amass_utils import NUM_KEYPT_VERTS, CONTACT_INDS
from losses.humor_loss import CONTACT_THRESH

NUM_WORKERS = 0

def parse_args(argv):
    # create config and parse args
    config = TestConfig(argv)
    known_args, unknown_args = config.parse()
    print('Unrecognized args: ' + str(unknown_args))
    return known_args

def test(args_obj, config_file):

    # set up output
    args = args_obj.base
    mkdir(args.out)

    # create logging system
    test_log_path = os.path.join(args.out, 'test.log')
    Logger.init(test_log_path)

    # save arguments used
    Logger.log('Base args: ' + str(args))
    Logger.log('Model args: ' + str(args_obj.model))
    Logger.log('Dataset args: ' + str(args_obj.dataset))
    Logger.log('Loss args: ' + str(args_obj.loss))

    # save training script/model/dataset/config used
    test_scripts_path = os.path.join(args.out, 'test_scripts')
    mkdir(test_scripts_path)
    pkg_root = os.path.join(cur_file_path, '..')
    dataset_file = class_name_to_file_name(args.dataset)
    dataset_file_path = os.path.join(pkg_root, 'datasets/' + dataset_file + '.py')
    model_file = class_name_to_file_name(args.model)
    loss_file = class_name_to_file_name(args.loss)
    model_file_path = os.path.join(pkg_root, 'models/' + model_file + '.py')
    train_file_path = os.path.join(pkg_root, 'test/test_humor.py')
    cp_files(test_scripts_path, [train_file_path, model_file_path, dataset_file_path, config_file])

    # load model class and instantiate
    model_class = importlib.import_module('models.' + model_file)
    Model = getattr(model_class, args.model)
    model = Model(**args_obj.model_dict,
                    model_smpl_batch_size=args.batch_size) # assumes model is HumorModel

    # load loss class and instantiate
    loss_class = importlib.import_module('losses.' + loss_file)
    Loss = getattr(loss_class, args.loss)
    loss_func = Loss(**args_obj.loss_dict,
                      smpl_batch_size=args.batch_size*args_obj.dataset.sample_num_frames) # assumes loss is HumorLoss

    device = get_device(args.gpu)
    model.to(device)
    loss_func.to(device)

    print(model)

    # count params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    Logger.log('Num model params: ' + str(params))

    # freeze params in loss
    for param in loss_func.parameters():
        param.requires_grad = False

    # load in pretrained weights if given
    if args.ckpt is not None:
        start_epoch, min_val_loss, min_train_loss = load_state(args.ckpt, model, optimizer=None, map_location=device, ignore_keys=model.ignore_keys)
        Logger.log('Successfully loaded saved weights...')
        Logger.log('Saved checkpoint is from epoch idx %d with min val loss %.6f...' % (start_epoch, min_val_loss))
    else:
        Logger.log('ERROR: No weight specified to load!!')
        # return

    # load dataset class and instantiate training and validation set
    if args.test_on_train:
        Logger.log('WARNING: running evaluation on TRAINING data as requested...should only be used for debugging!')
    elif args.test_on_val:
        Logger.log('WARNING: running evaluation on VALIDATION data as requested...should only be used for debugging!')
    Dataset = getattr(importlib.import_module('datasets.' + dataset_file), args.dataset)
    split = 'test'
    if args.test_on_train:
        split = 'train'
    elif args.test_on_val:
        split = 'val'
    test_dataset = Dataset(split=split, **args_obj.dataset_dict)
    # create loaders
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size,
                            shuffle=args.shuffle_test, 
                            num_workers=NUM_WORKERS,
                            pin_memory=True,
                            drop_last=False,
                            worker_init_fn=lambda _: np.random.seed())

    test_dataset.return_global = True
    model.dataset = test_dataset

    if args.eval_full_test:
        Logger.log('Running full test set evaluation...')
        # stats tracker
        tensorboard_path = os.path.join(args.out, 'test_tensorboard')
        mkdir(tensorboard_path)
        stat_tracker = StatTracker(tensorboard_path)

        # testing with same stats as training
        test_start_t = time.time()
        test_dataset.pre_batch()
        model.eval()
        for i, data in enumerate(test_loader):
            batch_start_t = time.time()
            # run model
            #   note we're always using ground truth input so this is only measuring single-step error, just like in training
            loss, stats_dict = model_class.step(model, loss_func, data, test_dataset, device, 0, mode='test', use_gt_p=1.0)

            # collect stats
            batch_elapsed_t = time.time() - batch_start_t
            total_elapsed_t = time.time() - test_start_t
            stats_dict['loss'] = loss
            stats_dict['time_per_batch'] = torch.Tensor([batch_elapsed_t])[0]

            stat_tracker.update(stats_dict, tag='test')

            if i % args.print_every == 0:
                stat_tracker.print(i, len(test_loader),
                                0, 1,
                                total_elapsed_time=total_elapsed_t,
                                tag='test')

            test_dataset.pre_batch()

    if args.eval_sampling or args.eval_sampling_debug:
        eval_sampling(model, test_dataset, test_loader, device,
                            out_dir=args.out if args.eval_sampling else None,
                            num_samples=args.eval_num_samples,
                            samp_len=args.eval_sampling_len,
                            viz_contacts=args.viz_contacts,
                            viz_pred_joints=args.viz_pred_joints,
                            viz_smpl_joints=args.viz_smpl_joints)

    if args.eval_recon or args.eval_recon_debug:
        eval_recon(model, test_dataset, test_loader, device,
                            out_dir=args.out if args.eval_recon else None,
                            viz_contacts=args.viz_contacts,
                            viz_pred_joints=args.viz_pred_joints,
                            viz_smpl_joints=args.viz_smpl_joints)

    Logger.log('Finished!')

def eval_sampling(model, test_dataset, test_loader, device, 
                  out_dir=None,
                  num_samples=1,
                  samp_len=10.0,
                  viz_contacts=False,
                  viz_pred_joints=False,
                  viz_smpl_joints=False):
    Logger.log('Evaluating sampling qualitatively...')
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    eval_qual_samp_len = int(samp_len * 30.0) # at 30 Hz

    res_out_dir = None
    if out_dir is not None:
        res_out_dir = os.path.join(out_dir, 'eval_sampling')
        if not os.path.exists(res_out_dir):
            os.mkdir(res_out_dir)

    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS
    male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
    male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)
    female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=eval_qual_samp_len).to(device)

    with torch.no_grad():
        test_dataset.pre_batch()
        model.eval()
        for i, data in enumerate(test_loader):
            # get inputs
            batch_in, batch_out, meta = data
            print(meta['path'])
            seq_name_list = [spath[:-4] for spath in meta['path']]
            if res_out_dir is None:
                batch_res_out_list = [None]*len(seq_name_list)
            else:
                batch_res_out_list = [os.path.join(res_out_dir, seq_name.replace('/', '_') + '_b' + str(i) + 'seq' + str(sidx)) for sidx, seq_name in enumerate(seq_name_list)]
                print(batch_res_out_list)
            # continue
            x_past, _, gt_dict, input_dict, global_gt_dict = model.prepare_input(batch_in, device, 
                                                                                data_out=batch_out,
                                                                                return_input_dict=True,
                                                                                return_global_dict=True)

            # roll out predicted motion
            B, T, _, _ = x_past.size()
            x_past = x_past[:,0,:,:] # only need input for first step
            rollout_input_dict = dict()
            for k in input_dict.keys():
                rollout_input_dict[k] = input_dict[k][:,0,:,:] # only need first step

            # sample same trajectory multiple times and save the joints/contacts output
            for samp_idx in range(num_samples):
                x_pred_dict = model.roll_out(x_past, rollout_input_dict, eval_qual_samp_len, gender=meta['gender'], betas=meta['betas'].to(device))

                # visualize and save
                print('Visualizing sample %d/%d!' % (samp_idx+1, num_samples))
                imsize = (1080, 1080)
                cur_res_out_list = batch_res_out_list
                if res_out_dir is not None:
                    cur_res_out_list = [out_path + '_samp%d' % (samp_idx) for out_path in batch_res_out_list]
                    imsize = (720, 720)
                viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm, cur_res_out_list,
                                imw=imsize[0],
                                imh=imsize[1],
                                show_smpl_joints=viz_smpl_joints,
                                show_pred_joints=viz_pred_joints,
                                show_contacts=viz_contacts
                              )

def eval_recon(model, test_dataset, test_loader, device, 
                  out_dir=None,
                  num_samples=1,
                  samp_len=10.0,
                  viz_contacts=False,
                  viz_pred_joints=False,
                  viz_smpl_joints=False):
    Logger.log('Evaluating reconstruction qualitatively...')
    from body_model.body_model import BodyModel
    from body_model.utils import SMPLH_PATH

    res_out_dir = None
    if out_dir is not None:
        res_out_dir = os.path.join(out_dir, 'eval_recon')
        if not os.path.exists(res_out_dir):
            os.mkdir(res_out_dir)

    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS
    male_bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    female_bm_path = os.path.join(SMPLH_PATH, 'female/model.npz')
    male_bm = BodyModel(bm_path=male_bm_path, num_betas=16, batch_size=test_dataset.sample_num_frames).to(device)
    female_bm = BodyModel(bm_path=female_bm_path, num_betas=16, batch_size=test_dataset.sample_num_frames).to(device)

    with torch.no_grad():
        test_dataset.pre_batch()
        model.eval()
        for i, data in enumerate(test_loader):
            # get inputs
            batch_in, batch_out, meta = data
            print(meta['path'])
            seq_name_list = [spath[:-4] for spath in meta['path']]
            if res_out_dir is None:
                batch_res_out_list = [None]*len(seq_name_list)
            else:
                batch_res_out_list = [os.path.join(res_out_dir, seq_name.replace('/', '_') + '_b' + str(i) + 'seq' + str(sidx)) for sidx, seq_name in enumerate(seq_name_list)]
                print(batch_res_out_list)

            _, _, _, global_gt_dict = model.prepare_input(batch_in, device, 
                                                            data_out=batch_out,
                                                            return_input_dict=False,
                                                            return_global_dict=True)
            
            # # NOTE: DEBUG add random translation to ensure canonicalization is properly handled in infer_global_seq and roll_out
            global_gt_dict['trans'] += torch.tensor([5.0, 5.0, 0.0]).reshape((1,1,1,3)).to(global_gt_dict['trans'])
            global_gt_dict['joints'] += torch.tensor([5.0, 5.0, 0.0]).reshape((1,1,1,3)).expand((1,1,22,3)).reshape((1,1,1,66)).to(global_gt_dict['joints'])

            # model doesn't take in contacts, only ['trans', 'trans_vel', 'root_orient', 'root_orient_vel', 'pose_body', 'joints', 'joints_vel']
            #   also don't need extra dimension that loader gives by default.
            global_in_dict = {k : v[:,:,0].clone() for k,v in global_gt_dict.items() if k != 'contacts'}

            # Encode. i.e. infer latent z vector for all pairs of frames in the seq
            #           Note this function can handle arbitrary "global" sequences
            #           (i.e. the first frame doesn't have to be in canonical system already, it will do this internally)
            encode_results = model.infer_global_seq(global_in_dict)
            prior_z_out, posterior_z_out = encode_results
            latent_z_seq = posterior_z_out[0] # use mean of the posterior (encoder) output
            # Decode. roll out reconstructed motion starting from initial step using the latent z sequence
            #   for this, only need the step at t=0, but do need the extra dimension, i.e. should be size (B, 1, D)
            decode_input_dict = {k : v[:,0].clone() for k, v in global_gt_dict.items() if k != 'contacts'} 
            # canonicalize_input=True allows it to handle any "global" inial state input (i.e. doesn't need to be in canonical frame already)
            #       and uncanonicalize_output will transform back into the input "global" frame
            x_pred_dict = model.roll_out(None, decode_input_dict, latent_z_seq.size(1),
                                         z_seq=latent_z_seq,
                                         gender=meta['gender'],
                                         betas=meta['betas'].to(device),
                                         canonicalize_input=True,
                                         uncanonicalize_output=True) 

            # assemble full reconstruction (initial state + decoded states)
            recon_pred_dict = {k : torch.cat([v[:,0:1,0], x_pred_dict[k]], dim=1) for k, v in global_gt_dict.items()}

            # visualize and save
            print('Visualizing ground truth!')
            imsize = (1080, 1080)
            cur_res_out_list = batch_res_out_list
            if res_out_dir is not None:
                cur_res_out_list = [out_path + '_gt' for out_path in batch_res_out_list]
                imsize = (720, 720)
            viz_gt_dict = {k : v[:,:,0] for k,v in global_gt_dict.items()}
            viz_eval_samp(global_gt_dict, viz_gt_dict, meta, male_bm, female_bm, cur_res_out_list,
                            imw=imsize[0],
                            imh=imsize[1],
                            show_smpl_joints=viz_smpl_joints,
                            show_pred_joints=viz_pred_joints,
                            show_contacts=viz_contacts
                            )
            print('Visualizing reconstruction!')
            cur_res_out_list = batch_res_out_list
            if res_out_dir is not None:
                cur_res_out_list = [out_path + '_recon' for out_path in batch_res_out_list]
            viz_eval_samp(global_gt_dict, recon_pred_dict, meta, male_bm, female_bm, cur_res_out_list,
                            imw=imsize[0],
                            imh=imsize[1],
                            show_smpl_joints=viz_smpl_joints,
                            show_pred_joints=viz_pred_joints,
                            show_contacts=viz_contacts
                            )

def viz_eval_samp(global_gt_dict, x_pred_dict, meta, male_bm, female_bm, out_path_list,
                    imw=720,
                    imh=720,
                    show_pred_joints=False,
                    show_smpl_joints=False,
                    show_contacts=False):
    '''
    Given x_pred_dict from the model rollout and the ground truth dict, runs through SMPL model to visualize
    '''
    J = len(SMPL_JOINTS)
    V = NUM_KEYPT_VERTS

    pred_world_root_orient = x_pred_dict['root_orient']
    B, T, _ = pred_world_root_orient.size()
    pred_world_root_orient = rotation_matrix_to_angle_axis(pred_world_root_orient.reshape((B*T, 3, 3))).reshape((B, T, 3))
    pred_world_pose_body = x_pred_dict['pose_body']
    pred_world_pose_body = rotation_matrix_to_angle_axis(pred_world_pose_body.reshape((B*T*(J-1), 3, 3))).reshape((B, T, (J-1)*3))
    pred_world_trans = x_pred_dict['trans']
    pred_world_joints = x_pred_dict['joints'].reshape((B, T, J, 3))

    viz_contacts = [None]*B
    if show_contacts and 'contacts' in x_pred_dict.keys():
        pred_contacts = torch.sigmoid(x_pred_dict['contacts'])
        pred_contacts = (pred_contacts > CONTACT_THRESH).to(torch.float)
        viz_contacts = torch.zeros((B, T, len(SMPL_JOINTS))).to(pred_contacts)
        viz_contacts[:,:,CONTACT_INDS] = pred_contacts
        pred_contacts = viz_contacts

    betas = meta['betas'].to(global_gt_dict[list(global_gt_dict.keys())[0]].device)
    for b in range(B):
        bm_world = male_bm if meta['gender'][b] == 'male' else female_bm
        # pred
        body_pred = bm_world(pose_body=pred_world_pose_body[b], 
                        pose_hand=None,
                        betas=betas[b,0].reshape((1, -1)).expand((T, 16)),
                        root_orient=pred_world_root_orient[b],
                        trans=pred_world_trans[b])

        pred_smpl_joints = body_pred.Jtr[:, :J]
        viz_joints = None
        if show_smpl_joints:
            viz_joints = pred_smpl_joints
        elif show_pred_joints:
            viz_joints = pred_world_joints[b]

        cur_offscreen = out_path_list[b] is not None
        from viz.utils import viz_smpl_seq, create_video
        body_alpha = 0.5 if viz_joints is not None and cur_offscreen else 1.0
        viz_smpl_seq(body_pred,
                        imw=imw, imh=imh, fps=30,
                        render_body=True,
                        render_joints=viz_joints is not None,
                        render_skeleton=viz_joints is not None and cur_offscreen,
                        render_ground=True,
                        contacts=viz_contacts[b],
                        joints_seq=viz_joints,
                        body_alpha=body_alpha,
                        use_offscreen=cur_offscreen,
                        out_path=out_path_list[b],
                        wireframe=False,
                        RGBA=False,
                        follow_camera=True,
                        cam_offset=[0.0, 2.2, 0.9],
                        joint_color=[ 0.0, 1.0, 0.0 ],
                        point_color=[0.0, 0.0, 1.0],
                        skel_color=[0.5, 0.5, 0.5],
                        joint_rad=0.015,
                        point_rad=0.015
                )

        if cur_offscreen:
            create_video(out_path_list[b] + '/frame_%08d.' + '%s' % ('png'), out_path_list[b] + '.mp4', 30)


def main(args, config_file):
    test(args, config_file)

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    config_file = sys.argv[1:][0][1:]
    main(args, config_file)