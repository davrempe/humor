import sys, os
file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '.'))

import argparse, importlib

class SplitLineParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()

class Args():
    '''
    Container class to hold parsed arguments for the base/train/test configuration along with
    model and dataset-specific.
    '''
    def __init__(self, base, model=None, dataset=None, loss=None):
        self.base = base
        self.model = model
        self.dataset = dataset
        self.loss = loss

        # dictionary versions of args that can be used to pass as constructor arguments
        self.model_dict = vars(self.model) if self.model is not None else None
        self.dataset_dict = vars(self.dataset) if self.dataset is not None else None
        self.loss_dict = vars(self.loss) if self.loss is not None else None

class BaseConfig():
    '''
    Base configuration, arguments apply to both training and evaluation scripts.
    This configuration will automatically load the sub-configuration of the specified model and dataset if available.
    '''
    def __init__(self, argv):
        self.argv = argv
        self.parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

        self.parser.add_argument('--dataset', type=str, required=True, choices=['AmassDiscreteDataset'], help='The name of the dataset type.')
        self.parser.add_argument('--model', type=str, required=True, help='The name of the model to use.')
        self.parser.add_argument('--loss', type=str, default=None, help='The name of the loss to use.')
        self.parser.add_argument('--out', type=str, default='./output', help='The directory to save outputs to (logs, results, weights, etc..).')
        self.parser.add_argument('--ckpt', type=str, default=None, help='Path to model weights to start training/testing from.')

        self.parser.add_argument('--gpu', type=int, default=0, help='The GPU index to use.')

        self.parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training.')
        self.parser.add_argument('--print-every', type=int, default=1, help='Number of batches between printing stats.')


    def parse(self):
        base_args, unknown_args = self.parser.parse_known_args(self.argv)

        # load any model-specific configuration
        model_args = None
        try:
            ModelConfig = getattr(importlib.import_module('config'), base_args.model + 'Config')
            model_config = ModelConfig(self.argv)
            model_args, model_unknown_args = model_config.parse()
        except AttributeError:
            print('No model-specific configuration for %s...' % (base_args.model))
            model_unknown_args = unknown_args

        # load any dataset-specific configuration
        dataset_args = None
        try:
            DataConfig = getattr(importlib.import_module('config'), base_args.dataset + 'Config')
            data_config = DataConfig(self.argv)
            dataset_args, data_unknown_args = data_config.parse()
        except AttributeError:
            print('No data-specific configuration for %s...' % (base_args.dataset))
            data_unknown_args = unknown_args

        # load any dataset-specific configuration
        use_loss_config = base_args.loss is not None
        loss_args = None
        loss_unknown_args = []
        if use_loss_config:
            loss_args = None
            try:
                LossConfig = getattr(importlib.import_module('config'), base_args.loss + 'Config')
                loss_config = LossConfig(self.argv)
                loss_args, loss_unknown_args = loss_config.parse()
            except AttributeError:
                print('No data-specific configuration for %s...' % (base_args.loss))
                loss_unknown_args = unknown_args

        # make sure unknown args are unknown to both if returning
        unknown_args = set([arg for arg in unknown_args if arg[:2] == '--'])
        model_unknown_args = set([arg for arg in model_unknown_args if arg[:2] == '--'])
        data_unknown_args = set([arg for arg in data_unknown_args if arg[:2] == '--'])
        loss_unknown_args = set([arg for arg in loss_unknown_args if arg[:2] == '--'])
        final_unknown_args = list(unknown_args.intersection(model_unknown_args, data_unknown_args, loss_unknown_args))

        final_args = Args(base_args, model=model_args, dataset=dataset_args, loss=loss_args)

        return final_args, final_unknown_args

    # default args: model name (load additional args based on specific model name), dataset name

class BaseSubConfig():
    '''
    Base sub-configuration, each model/dataset-specific sub-configuration should derive from this.
    '''
    def __init__(self, argv):
        self.argv = argv
        self.parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    def parse(self, namespace=None):
        self.args = self.parser.parse_known_args(self.argv, namespace=namespace)
        return self.args

#
# NOTE: Edit/Add these configs for changes in training and testing scripts
#

class TrainConfig(BaseConfig):
    def __init__(self, argv):
        super(TrainConfig, self).__init__(argv)

        self.parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training.')
        self.parser.add_argument('--val-every', type=int, default=1, help='Number of epochs between validations.')
        self.parser.add_argument('--save-every', type=int, default=1, help='Number of epochs between saving model checkpoints.')

        self.parser.add_argument('--lr', type=float, default=1e-3, help='Starting learning rate.')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for ADAM')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 for ADAM')
        self.parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon rate for ADAM')
        self.parser.add_argument('--sched-milestones', type=int, nargs='+', default=[1], help='List of epochs to decay learning rate.')
        self.parser.add_argument('--sched-decay', type=float, default=1.0, help='The decay rate of the LR scheduler, by default there is no decay.')

        self.parser.add_argument('--decay', type=float, default=0.0, help='Weight decay on params.')

        self.parser.add_argument('--no-load-optim', dest='load_optim', action='store_false', help="If given, will not load the state of the optimizer to continue training from a chekcpoint.")
        self.parser.set_defaults(load_optim=True)

        self.parser.add_argument('--adam', dest='use_adam', action='store_true', help="If given, uses Adam optimizer rather than Adamax.")
        self.parser.set_defaults(use_adam=False)

        self.parser.add_argument('--sched-samp-start', type=int, default=None, help='The epoch at which to start scheduled sampling after the supervised phase of training.')
        self.parser.add_argument('--sched-samp-end', type=int, default=None, help='The epoch at which to end scheduled sampling which moves on to the autoregressive phase of training.')

class TestConfig(BaseConfig):
    def __init__(self, argv):
        super(TestConfig, self).__init__(argv)
        # NOTE: add test-specific options here (e.g. which test to run)
        self.parser.add_argument('--shuffle-test', dest='shuffle_test', action='store_true', help="Shuffles test data.")
        self.parser.set_defaults(shuffle_test=False)
        self.parser.add_argument('--test-on-train', dest='test_on_train', action='store_true', help="Runs evaluation on TRAINING data.")
        self.parser.set_defaults(test_on_train=False)
        self.parser.add_argument('--test-on-val', dest='test_on_val', action='store_true', help="Runs evaluation on VALIADTION data.")
        self.parser.set_defaults(test_on_val=False)

        self.parser.add_argument('--eval-sampling', dest='eval_sampling', action='store_true', help="Visualizing random sample rollouts")
        self.parser.set_defaults(eval_sampling=False)
        self.parser.add_argument('--eval-sampling-len', type=float, default=10.0, help='Number of seconds to sample for (default 10 s)')
        self.parser.add_argument('--eval-sampling-debug', dest='eval_sampling_debug', action='store_true', help="Visualizes random samples in interactive visualization.")
        self.parser.set_defaults(eval_sampling_debug=False)
        self.parser.add_argument('--eval-test', dest='eval_full_test', action='store_true', help="Evaluate on the full test set with same metrics as during training.")
        self.parser.set_defaults(eval_full_test=False)
        self.parser.add_argument('--eval-num-samples', type=int, default=1, help='Number of times to sample the model for the same initial state for eval_sampling evalutations.')
        self.parser.add_argument('--eval-recon', dest='eval_recon', action='store_true', help="Visualizes reconstructions of random AMASS sequences")
        self.parser.add_argument('--eval-recon-debug', dest='eval_recon_debug', action='store_true', help="Interactively visualizes reconstructions of random AMASS sequences")

        self.parser.add_argument('--viz-contacts', dest='viz_contacts', action='store_true', help="For visualization, body mesh is translucent and contacts are shown on SMPL joint skeleton.")
        self.parser.set_defaults(viz_contacts=False)
        self.parser.add_argument('--viz-pred-joints', dest='viz_pred_joints', action='store_true', help="For visualization, HuMoR output joints are visualized.")
        self.parser.set_defaults(viz_pred_joints=False)
        self.parser.add_argument('--viz-smpl-joints', dest='viz_smpl_joints', action='store_true', help="For visualization, SMPL joints are visualized (determined from HuMoR output joint angles).")
        self.parser.set_defaults(viz_smpl_joints=False)

#
# Edit/add configs here for changes to model-specific arguments.
# NOTE: must be named ModelNameConfig to be properly loaded. Also should not clash names with any Base/Train/Test configuration flags.
#

class HumorModelConfig(BaseSubConfig):
    '''
    Configuration for arguments specific to models.HumorModel model class.
    '''
    def __init__(self, argv):
        super(HumorModelConfig, self).__init__(argv)
        # arguments specific to this model
        self.parser.add_argument('--out-rot-rep', type=str, default='aa', choices=['aa', '6d', '9d'], help='Rotation representation to output from the model.')
        self.parser.add_argument('--in-rot-rep', type=str, default='mat', choices=['aa', '6d', 'mat'], help='Rotation representation to input to the model for the relative full sequence input.')
        self.parser.add_argument('--latent-size', type=int, default=48, help='Size of the latent feature.')

        self.parser.add_argument('--model-steps-in', dest='steps_in', type=int, default=1, help='At each step of the sequence, the number of input frames.')

        self.parser.add_argument('--no-conditional-prior', dest='conditional_prior', action='store_false', help="Conditions the prior on the past input sequence.")
        self.parser.set_defaults(conditional_prior=True)
        self.parser.add_argument('--no-output-delta', dest='output_delta', action='store_false', help="Each step predicts the residual rather than the next step.")
        self.parser.set_defaults(output_delta=True)

        self.parser.add_argument('--posterior-arch', type=str, default='mlp', choices=['mlp'], help='')
        self.parser.add_argument('--decoder-arch', type=str, default='mlp', choices=['mlp'], help='')
        self.parser.add_argument('--prior-arch', type=str, default='mlp', choices=['mlp'], help='')

        self.parser.add_argument('--model-data-config', type=str, default='smpl+joints+contacts', choices=['smpl+joints', 'smpl+joints+contacts'], help='which state configuration to use for the model')

        self.parser.add_argument('--no-detach-sched-samp', dest='detach_sched_samp', action='store_false', help="Allows gradients to backprop through multiple output steps when using schedules sampling.")
        self.parser.set_defaults(detach_sched_samp=True)

        self.parser.add_argument('--model-use-smpl-joint-inputs', dest='model_use_smpl_joint_inputs', action='store_true', help="uses smpl joints rather than regressed joints to input at next step (during rollout and sched samp).")
        self.parser.set_defaults(model_use_smpl_joint_inputs=False)


# Edit/add configs here for changes to dataset-specific arguments.
# NOTE: must be named DatasetNameConfig to be properly loaded. Also should not clash names with any Base/Train/Test configuration flags.
#

class AmassDiscreteDatasetConfig(BaseSubConfig):
    '''
    Configuration for arguments specific to models.AmassDiscreteDataset dataset class.
    '''
    def __init__(self, argv):
        super(AmassDiscreteDatasetConfig, self).__init__(argv)
        # arguments specific to this dataset
        self.parser.add_argument('--data-paths', type=str, nargs='+', required=True, help='Paths to dataset roots.')
        self.parser.add_argument('--split-by', type=str, default='dataset', choices=['single', 'sequence', 'subject', 'dataset'], help='How to split the dataset into train/test/val.')
        self.parser.add_argument('--splits-path', type=str, default=None, help='Path to data splits to use.')
        self.parser.add_argument('--sample-num-frames', type=int, default=10, help=' the number of frames returned for each sequence, i.e. the number of input/output pairs.')
        self.parser.add_argument('--data-rot-rep', type=str, default='mat', choices=['aa', 'mat', '6d'], help='the rotation representation for the INPUT data. [aa, mat, 6d] Output data is always given as a rotation matrix.')

        self.parser.add_argument('--data-steps-in', dest='step_frames_in', type=int, default=1, help='At each step of the sequence, the number of input frames.')
        self.parser.add_argument('--data-steps-out', dest='step_frames_out', type=int, default=1, help='At each step of the sequence, the number of output frames.')
        self.parser.add_argument('--data-out-step-size', dest='frames_out_step_size', type=int, default=1, help='Spacing between the output frames.')

        self.parser.add_argument('--data-return-config', type=str, default='smpl+joints+contacts', choices=['smpl+joints', 'smpl+joints+contacts', 'all'], help='which values to return from the data loader')
        self.parser.add_argument('--data-noise-std', type=float, default=0.0, help='Standard deviation for gaussian noise to add to input motion.')

class HumorLossConfig(BaseSubConfig):
    '''
    Configuration for arguments specific to losses.HumorLoss dataset class.
    '''
    def __init__(self, argv):
        super(HumorLossConfig, self).__init__(argv)

        self.parser.add_argument('--kl-loss', type=float, default=0.0004, help='Loss weight')
        self.parser.add_argument('--kl-loss-anneal-start', type=int, default=0, help='The epoch that the kl loss will start linearly increasing from 0.0')
        self.parser.add_argument('--kl-loss-anneal-end', type=int, default=50, help='The epoch that the kl loss will reach its full weight')
        self.parser.add_argument('--kl-loss-cycle-len', type=int, default=-1, help='If > 0, KL annealing will be done cyclicly and it will last this many epochs per cycle. If given will ignore kl-loss-anneal-start/end.')

        self.parser.add_argument('--regr-trans-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-trans-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-root-orient-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-root-orient-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-pose-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-pose-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-joint-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-joint-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-joint-orient-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-vert-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--regr-vert-vel-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--contacts-loss', type=float, default=0.01, help='Loss weight')
        self.parser.add_argument('--contacts-vel-loss', type=float, default=0.01, help='Loss weight')

        self.parser.add_argument('--smpl-joint-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--smpl-mesh-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--smpl-joint-consistency-loss', type=float, default=1.0, help='Loss weight')
        self.parser.add_argument('--smpl-vert-consistency-loss', type=float, default=0.0, help='Loss weight')