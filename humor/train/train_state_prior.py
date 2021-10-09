import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

import argparse, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.distributions import MixtureSameFamily, MultivariateNormal, Independent, Categorical, Normal 

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from datasets.amass_discrete_dataset import AmassDiscreteDataset

def parse_args(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--data', type=str, required=True, help='Path to the full AMASS dataset')
    parser.add_argument('--out', type=str, required=True, help='Path to save outputs to')
    parser.add_argument('--train-states', type=str, default=None, help='npy file with pre-loaded train states')
    parser.add_argument('--gmm-comps', type=int, default=12, help='Number of GMM components to use')

    parser.add_argument('--viz-only', dest='viz_only', action='store_true', help="If given, only visualizes results in the given output directory, does not refit.")
    parser.set_defaults(viz_only=False)
    parser.add_argument('--test-only', dest='test_only', action='store_true', help="If given, only runs fitting in the given output directory on test data, does not refit.")
    parser.set_defaults(test_only=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args

def main(args):
    print(args)

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    all_states_out_path = os.path.join(args.out, 'train_states.npy')
    gmm_out_path = os.path.join(args.out, 'prior_gmm.npz')
    if args.viz_only:
        print('Visualizing results...')
        viz_gmm_fit_results(gmm_out_path)
        exit()
    if args.test_only:
        print('Evaluating on test set...')
        test_results(args.data, gmm_out_path)
        exit()

    all_states = None
    if args.train_states is not None:
        start_t = time.time()
        print('Loading processed train states...')
        all_states = np.load(args.train_states)
        print('Loaded in %f s' % (time.time() - start_t))
    else:
        amass_dataset = AmassDiscreteDataset(split='train',
                                            data_paths=[args.data],
                                            split_by='dataset',
                                            sample_num_frames=1,
                                            step_frames_in=1,
                                            step_frames_out=0,
                                            data_rot_rep='aa',
                                            data_return_config='smpl+joints',
                                            deterministic_train=True,
                                            return_global=False,
                                            only_global=False)

        batch_size = 1000
        loader = DataLoader(amass_dataset, 
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=8,
                        pin_memory=False,
                        drop_last=False,
                        worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

        all_states = []
        for i, data in enumerate(loader):
            print('Batch %d/%d...' % (i, len(loader)))
            start_t = time.time()
            batch_in, _, meta = data
            B = batch_in['joints'].size(0)
            joints = batch_in['joints'][:,0,0].reshape((B, -1))
            joints_vel = batch_in['joints_vel'][:,0,0].reshape((B, -1))
            trans_vel = batch_in['trans_vel'][:,0,0]
            root_orient_vel = batch_in['root_orient_vel'][:,0,0]

            cur_state = torch.cat([joints, joints_vel, trans_vel, root_orient_vel], dim=-1)
            all_states.append(cur_state)

        all_states = torch.cat(all_states, dim=0).numpy()
        np.save(all_states_out_path, all_states)

    print(all_states.shape)

    print('Fitting GMM with %d components...' % (args.gmm_comps))
    start_t = time.time()
    gmm = GaussianMixture(n_components=args.gmm_comps,
                         covariance_type='full',
                         tol=0.001,
                         reg_covar=1e-06,
                         max_iter=200,
                         n_init=1,
                         init_params='kmeans',
                         weights_init=None,
                         means_init=None,
                         precisions_init=None,
                         random_state=0,
                         warm_start=False,
                         verbose=1,
                         verbose_interval=5)
    gmm.fit(all_states)
    # print(gmm.weights_)
    # print(gmm.means_)
    # print(gmm.covariances_)
    print(gmm.converged_)
    print(gmm.weights_.shape)
    print(gmm.means_.shape)
    print(gmm.covariances_.shape)

    # save distirbution information
    np.savez(gmm_out_path, weights=gmm.weights_, means=gmm.means_, covariances=gmm.covariances_)

    print('GMM time: %f s' % (time.time() - start_t))

    print('Running evaluation on test set...')
    test_results(args.data, gmm_out_path)
    # print('Visualizing sampled results...')
    # viz_gmm_fit_results(gmm_out_path, debug_gmm_obj=gmm, debug_data=all_states)

def load_gmm_results(gmm_path):
    gmm_res = np.load(gmm_path)
    gmm_weights = gmm_res['weights']
    gmm_means = gmm_res['means']
    gmm_covs = gmm_res['covariances']
    return gmm_weights, gmm_means, gmm_covs

def build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs):
    mix = Categorical(torch.from_numpy(gmm_weights))
    comp = MultivariateNormal(torch.from_numpy(gmm_means), covariance_matrix=torch.from_numpy(gmm_covs))
    gmm_distrib = MixtureSameFamily(mix, comp)
    return gmm_distrib

def viz_gmm_fit_results(gmm_path, debug_gmm_obj=None, debug_data=None):

    # load in GMM result
    gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)

    # build pytorch distrib
    gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)
    print([gmm_distrib.batch_shape, gmm_distrib.event_shape])

    if debug_gmm_obj is not None and debug_data is not None:
        print('pytorch logprob...')
        torch_logprob = gmm_distrib.log_prob(torch.from_numpy(debug_data))
        print(torch_logprob.size())
        print(torch_logprob[:20])

        print('sklearn logprob...')
        sk_logprob = debug_gmm_obj.score_samples(debug_data)
        print(sk_logprob.shape)
        print(sk_logprob[:20])

    # sample randomly
    num_samps = 100
    sample_states = gmm_distrib.sample(torch.Size([num_samps]))

    num_samps = gmm_means.shape[0]
    sample_states = torch.from_numpy(gmm_means)
    print(gmm_weights)

    torch_logprob = gmm_distrib.log_prob(sample_states)

    print(sample_states.size())
    print(torch_logprob)
    print(torch_logprob.mean())

    # visualize joints and velocities
    from viz.utils import viz_results, viz_smpl_seq

    # visualize results
    viz_joints = sample_states[:,:66].reshape((num_samps, 22, 3))
    viz_joints_vel = sample_states[:,66:132].reshape((num_samps, 22, 3))
    viz_trans_vel = sample_states[:,132:135]
    viz_root_orient_vel = sample_states[:,135:]
    print(viz_joints.shape)
    print(viz_joints_vel.shape)
    print(viz_trans_vel.shape)
    print(viz_root_orient_vel.shape)
    print('Showing joint velocities...')
    viz_smpl_seq(None, imw=1080, imh=1080, fps=10, contacts=None,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    joints_seq=viz_joints,
                    joints_vel=viz_joints_vel)
    print('Showing root velocity...')
    viz_smpl_seq(None, imw=1080, imh=1080, fps=10, contacts=None,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    joints_seq=viz_joints,
                    joints_vel=viz_trans_vel.reshape((-1, 1, 3)).repeat((1, 22, 1)))
    print('Showing root orient velocity...')
    viz_smpl_seq(None, imw=1080, imh=1080, fps=10, contacts=None,
                    render_body=False, render_joints=True, render_skeleton=True, render_ground=True,
                    joints_seq=viz_joints,
                    joints_vel=viz_root_orient_vel.reshape((-1, 1, 3)).repeat((1, 22, 1)))

def test_results(data_path, gmm_path):
    #
    # Evaluate likelihood of test data
    #

    # load in GMM result
    gmm_weights, gmm_means, gmm_covs = load_gmm_results(gmm_path)

    # build pytorch distrib
    gmm_distrib = build_pytorch_gmm(gmm_weights, gmm_means, gmm_covs)

    # load in all test data
    test_dataset = AmassDiscreteDataset(split='test',
                                        data_paths=[data_path],
                                        split_by='dataset',
                                        sample_num_frames=1,
                                        step_frames_in=1,
                                        step_frames_out=0,
                                        data_rot_rep='aa',
                                        data_return_config='smpl+joints',
                                        deterministic_train=True,
                                        return_global=False,
                                        only_global=False)

    batch_size = 1000
    test_loader = DataLoader(test_dataset, 
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=False,
                    drop_last=False,
                    worker_init_fn=lambda _: np.random.seed()) # get around numpy RNG seed bug

    test_states = []
    for i, data in enumerate(test_loader):
        print('Batch %d/%d...' % (i, len(test_loader)))
        start_t = time.time()
        batch_in, _, meta = data
        # print(meta['path'])

        B = batch_in['joints'].size(0)
        joints = batch_in['joints'][:,0,0].reshape((B, -1))
        joints_vel = batch_in['joints_vel'][:,0,0].reshape((B, -1))
        trans_vel = batch_in['trans_vel'][:,0,0]
        root_orient_vel = batch_in['root_orient_vel'][:,0,0]

        cur_state = torch.cat([joints, joints_vel, trans_vel, root_orient_vel], dim=-1)
        test_states.append(cur_state)

    test_states = torch.cat(test_states, dim=0)
    print(test_states.size())

    # eval likelihood
    test_logprob = gmm_distrib.log_prob(test_states)
    mean_logprob = test_logprob.mean()

    print('Mean test logprob: %f' % (mean_logprob.item()))

if __name__=='__main__':
    args = parse_args(sys.argv[1:])
    main(args)