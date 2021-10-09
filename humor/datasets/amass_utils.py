
from body_model.utils import SMPL_JOINTS


TRAIN_DATASETS = ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje', 'BMLmovi', 
                    'EKUT', 'ACCAD']
TEST_DATASETS = ['Transitions_mocap', 'HumanEva']
VAL_DATASETS = ['MPI_HDM05', 'SFU', 'MPI_mosh']


SPLITS = ['train', 'val', 'test', 'custom']
SPLIT_BY = [ 
             'single',   # the data path is a single .npz file. Don't split: train and test are same
             'sequence', # the data paths are directories of subjects. Collate and split by sequence.
             'subject',  # the data paths are directories of datasets. Collate and split by subject.
             'dataset'   # a single data path to the amass data root is given. The predefined datasets will be used for each split.
            ]

ROT_REPS = ['mat', 'aa', '6d']

# these correspond to [root, left knee, right knee, left heel, right heel, left toe, right toe, left hand, right hand]
CONTACT_ORDERING = ['hips', 'leftLeg', 'rightLeg', 'leftFoot', 'rightFoot', 'leftToeBase', 'rightToeBase', 'leftHand', 'rightHand']
CONTACT_INDS = [SMPL_JOINTS[jname] for jname in CONTACT_ORDERING]

NUM_BODY_JOINTS = len(SMPL_JOINTS)-1
NUM_KEYPT_VERTS = 43

DATA_NAMES = ['trans', 'trans_vel', 'root_orient', 'root_orient_vel', 'pose_body', 'pose_body_vel', 'joints', 'joints_vel', 'joints_orient_vel', 'verts', 'verts_vel', 'contacts']

SMPL_JOINTS_RETURN_CONFIG = {
    'trans' : True,
    'trans_vel' : True,
    'root_orient' : True,
    'root_orient_vel' : True,
    'pose_body' : True,
    'pose_body_vel' : False,
    'joints' : True,
    'joints_vel' : True,
    'joints_orient_vel' : False,
    'verts' : False,
    'verts_vel' : False,
    'contacts' : False
}

SMPL_JOINTS_CONTACTS_RETURN_CONFIG = {
    'trans' : True,
    'trans_vel' : True,
    'root_orient' : True,
    'root_orient_vel' : True,
    'pose_body' : True,
    'pose_body_vel' : False,
    'joints' : True,
    'joints_vel' : True,
    'joints_orient_vel' : False,
    'verts' : False,
    'verts_vel' : False,
    'contacts' : True
}

ALL_RETURN_CONFIG = {
    'trans' : True,
    'trans_vel' : True,
    'root_orient' : True,
    'root_orient_vel' : True,
    'pose_body' : True,
    'pose_body_vel' : False,
    'joints' : True,
    'joints_vel' : True,
    'joints_orient_vel' : False,
    'verts' : True,
    'verts_vel' : False,
    'contacts' : True
}

RETURN_CONFIGS = {
                  'smpl+joints+contacts' : SMPL_JOINTS_CONTACTS_RETURN_CONFIG,
                  'smpl+joints' : SMPL_JOINTS_RETURN_CONFIG,
                  'all' : ALL_RETURN_CONFIG
                  }

def data_name_list(return_config):
    '''
    returns the list of data values in the given configuration
    '''
    cur_ret_cfg = RETURN_CONFIGS[return_config]
    data_names = [k for k in DATA_NAMES if cur_ret_cfg[k]]
    return data_names

def data_dim(dname, rot_rep_size=9):
    '''
    returns the dimension of the data with the given name. If the data is a rotation, returns the size with the given representation.
    '''
    if dname in ['trans', 'trans_vel', 'root_orient_vel']:
        return 3
    elif dname in ['root_orient']:
        return rot_rep_size
    elif dname in ['pose_body']:
        return NUM_BODY_JOINTS*rot_rep_size
    elif dname in ['pose_body_vel']:
        return NUM_BODY_JOINTS*3
    elif dname in ['joints', 'joints_vel']:
        return len(SMPL_JOINTS)*3
    elif dname in ['joints_orient_vel']:
        return 1
    elif dname in ['verts', 'verts_vel']:
        return NUM_KEYPT_VERTS*3
    elif dname in ['contacts']:
        return len(CONTACT_ORDERING)
    else:
        print('The given data name %s is not valid!' % (dname))
        exit()