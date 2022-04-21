from utils.config import SplitLineParser

from fitting.fitting_utils import NSTAGES

def parse_args(argv):
    parser = SplitLineParser(fromfile_prefix_chars='@', allow_abbrev=False)

    # Observed data options
    parser.add_argument('--data-path', type=str, required=True, help='Path to the data to fit.')
    parser.add_argument('--data-type', type=str, required=True, choices=['AMASS', 'PROX-RGB', 'PROX-RGBD', 'iMapper-RGB', 'RGB'], help='The type of data we are fitting to.')
    parser.add_argument('--data-fps', type=int, default=30, help='Sampling rate of the data.')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of sequences to batch together for fitting to data.')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true', help="Shuffles data.")
    parser.set_defaults(shuffle=False)
    parser.add_argument('--op-keypts', type=str, default=None, help='(optional) path to a directory of custom detected OpenPose keypoints to use for RGB fitting rather than running OpenPose before optimization.')

    # AMASS-specific options
    parser.add_argument('--amass-split-by', type=str, default='dataset', choices=['single', 'sequence', 'subject', 'dataset'], help='How to split the dataset into train/test/val.')
    parser.add_argument('--amass-custom-split', type=str, nargs='+', default=None, help='Instead of using test set, use this custom list of datasets.')
    parser.add_argument('--amass-batch-size', type=int, default=-1, help='Number of sequences to batch together for fitting to AMASS data.')
    parser.add_argument('--amass-seq-len', type=int, default=60, help='Number of frames in AMASS sequences to fit.')
    parser.add_argument('--amass-use-joints', dest='amass_use_joints', action='store_true', help="Use 3D joint observations for fitting.")
    parser.set_defaults(amass_use_joints=False)
    parser.add_argument('--amass-root-joint-only', dest='amass_root_joint_only', action='store_true', help="Use 3D root joint observation for fitting.")
    parser.set_defaults(amass_root_joint_only=False)
    parser.add_argument('--amass-use-verts', dest='amass_use_verts', action='store_true', help="Use subset of 3D mesh vertices observations for fitting.")
    parser.set_defaults(amass_use_verts=False)
    parser.add_argument('--amass-use-points', dest='amass_use_points', action='store_true', help="Use sampled 3D points on mesh surface for fitting.")
    parser.set_defaults(amass_use_points=False)
    parser.add_argument('--amass-noise-std', type=float, default=0.0, help='Artificial gaussian noise standard deviation to add to observations.')
    parser.add_argument('--amass-make-partial', dest='amass_make_partial', action='store_true', help="Make the observations randomly partial.")
    parser.set_defaults(amass_make_partial=False)
    parser.add_argument('--amass-partial-height', type=float, default=0.9, help='Points/joints/verts under this z value will be dropped to make partial.')
    parser.add_argument('--amass-drop-middle', dest='amass_drop_middle', action='store_true', help="Drops the middle third frames from the sequence completely.")
    parser.set_defaults(amass_drop_middle=False)

    # PROX-specific options
    parser.add_argument('--prox-batch-size', type=int, default=-1, help='Number of sequences to batch together for fitting to PROX data.')
    parser.add_argument('--prox-seq-len', type=int, default=60, help='Number of frames in PROX sequences to fit.')
    parser.add_argument('--prox-recording', type=str, default=None, help='Fit to a specific PROX recording')
    parser.add_argument('--prox-recording-subseq-idx', type=int, default=-1, help='Fit to a specific PROX recording subsequence')

    # iMapper-specific options
    parser.add_argument('--imapper-seq-len', type=int, default=60, help='Number of frames in iMapper sequences to fit. ')
    parser.add_argument('--imapper-scene', type=str, default=None, help='Fit to a specific iMapper scene')
    parser.add_argument('--imapper-scene-subseq-idx', type=int, default=-1, help='Fit to a specific subsequence')

    # RGB-specific options
    parser.add_argument('--rgb-seq-len', type=int, default=None, help='If none, fits the whole video at once. If given, is the max number of frames to use when splitting the video into subseqeunces for fitting.')
    parser.add_argument('--rgb-overlap-len', type=int, default=None, help='If None, fitst the whole video at once. If given, is the minimum number of frames to overlap subsequences extracted from the given video. These overlapped frames are used in a consistency energy.')
    parser.add_argument('--rgb-intrinsics', type=str, default=None, help='Path to the camera intrinsics file to use for re-projection energy. If not given uses defaults.')
    parser.add_argument('--rgb-planercnn-res', type=str, default=None, help='Path to results of PlaneRCNN detection. If given uses this to initialize the floor plane otherwise uses defaults.')
    parser.add_argument('--rgb-overlap-consist-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Enforces consistency between overlapping subsequences within a batch in terms of the ground plane, shape params, and joint positions.')

    # PROX + iMapper + RGB options
    parser.add_argument('--mask-joints2d', dest='mask_joints2d', action='store_true', help="If true, masks the 2d joints based on the person segmentation occlusion mask.")
    parser.set_defaults(mask_joints2d=False)

    # Loss weights
    parser.add_argument('--joint3d-weight', type=float, nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints')
    parser.add_argument('--joint3d-rollout-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints from motion prior rollout.')
    parser.add_argument('--joint3d-smooth-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D joints differences')
    parser.add_argument('--vert3d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 3D verts')
    parser.add_argument('--point3d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Chamfer loss on 3D points')
    parser.add_argument('--joint2d-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 loss on 2D reprojection')
    parser.add_argument('--pose-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under pose prior')
    parser.add_argument('--shape-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under shape prior')
    parser.add_argument('--motion-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under motion prior')
    parser.add_argument('--init-motion-prior-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='likelihood under init state prior')
    parser.add_argument('--joint-consistency-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between SMPL and motion prior joints')
    parser.add_argument('--bone-length-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 difference between bone lengths of motion prior joints at consecutive frames.')
    parser.add_argument('--contact-vel-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Predicted contacting joints have 0 velocity when in contact')
    parser.add_argument('--contact-height-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='Predicted contacting joints are at the height of the floor')
    parser.add_argument('--floor-reg-weight', type=float,nargs=NSTAGES, default=[0.0, 0.0, 0.0], help='L2 Regularization that pushes floor to stay close to the initialization.')
    # loss options
    parser.add_argument('--robust-loss', type=str, default='bisquare', choices=['none', 'bisquare'], help='Which robust loss weighting to use for points3d losses (if any).')
    parser.add_argument('--robust-tuning-const', type=float, default=4.6851, help='Tuning constant to use in the robust loss.')
    parser.add_argument('--joint2d-sigma', type=float, default=100.0, help='scaling for robust geman-mclure function on joint2d.')

    # stage 3 options
    parser.add_argument('--stage3-no-tune-init-state', dest='stage3_tune_init_state', action='store_false', help="If given, will not use initial state tuning at the beginning of stage 3, instead optimizing full seq at once.")
    parser.set_defaults(stage3_tune_init_state=True)
    parser.add_argument('--stage3-tune-init-num-frames', type=int, default=15, help="When tuning initial state at the beginning of stage 3, uses this many initial frames.")
    parser.add_argument('--stage3-tune-init-freeze-start', type=int, default=30, help='Iteration to tune initial state until, at which point it is frozen and full latent sequence is optimized')
    parser.add_argument('--stage3-tune-init-freeze-end', type=int, default=55, help='Iteration to freeze initial state until, at which point full sequence and initial state are refined together.')
    parser.add_argument('--stage3-full-contact', dest='stage3_contact_refine_only', action='store_false', help="If given, uses contact losses for the entire stage 3 rather than just in the final refinement portion.")
    parser.set_defaults(stage3_contact_refine_only=True)

    # smpl model path
    parser.add_argument('--smpl', type=str, default='./body_models/smplh/neutral/model.npz', help='Path to SMPL model to use for optimization. Currently only SMPL+H is supported.')
    parser.add_argument('--gt-body-type', type=str, default='smplh', choices=['smplh'], help='Which body model to load in for GT data')
    parser.add_argument('--vposer', type=str, default='./body_models/vposer_v1_0', help='Path to VPoser checkpoint.')
    parser.add_argument('--openpose', type=str, default='./external/openpose', help='Path to OpenPose installation.')

    # motion prior weights and model information
    parser.add_argument('--humor', type=str, help='Path to HuMoR weights to use as the motion prior.')
    parser.add_argument('--humor-out-rot-rep', type=str, default='aa', choices=['aa', '6d', '9d'], help='Rotation representation to output from the model.')
    parser.add_argument('--humor-in-rot-rep', type=str, default='mat', choices=['aa', '6d', 'mat'], help='Rotation representation to input to the model for the relative full sequence input.')
    parser.add_argument('--humor-latent-size', type=int, default=48, help='Size of the latent feature.')
    parser.add_argument('--humor-model-data-config', type=str, default='smpl+joints+contacts', choices=['smpl+joints', 'smpl+joints+contacts'], help='which state configuration to use for the model')
    parser.add_argument('--humor-steps-in', type=int, default=1, help='Number of input timesteps the prior expects.')

    # init motion state prior information
    parser.add_argument('--init-motion-prior', type=str, default='./checkpoints/init_state_prior_gmm', help='Path to parameters of a GMM to use as the prior for initial motion state.')

    # optimization options
    parser.add_argument('--lr', type=float, default=1.0, help='step size during optimization')
    parser.add_argument('--num-iters', type=int, nargs=NSTAGES, default=[30, 80, 70], help='The number of optimization iterations at each stage (3 stages total)')
    parser.add_argument('--lbfgs-max-iter', type=int, default=20, help='The number of max optim iterations per LBFGS step.')

    # options to save/visualize results
    parser.add_argument('--out', type=str, default=None, help='Output path to save fitting results/visualizations to.')

    parser.add_argument('--save-results', dest='save_results', action='store_true', help="Saves final optimized and GT smpl results and observations")
    parser.set_defaults(save_results=False)
    parser.add_argument('--save-stages-results', dest='save_stages_results', action='store_true', help="Saves intermediate optimized results")
    parser.set_defaults(save_stages_results=False)

    known_args, unknown_args = parser.parse_known_args(argv)

    return known_args