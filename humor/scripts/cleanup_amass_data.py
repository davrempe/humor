import glob
import os
import argparse
import time
import random
import shutil

# which sub-datasets to clean up
BML_NTroje = True
MPI_HDM05 = True

def main(args):
    if not os.path.exists(args.data) or not os.path.isdir(args.data):
        print('Could not find path or it is not a directory')
        return

    if BML_NTroje:
        # remove treadmill clips from BML_NTroje
        # contains treadmill_
        # contains normal_
        dataset_path = os.path.join(args.data, 'BioMotionLab_NTroje')
        if not os.path.exists(dataset_path):
            print('Could not find BioMotionLab_NTroje data, not filtering out treadmill sequences...')
        else:
            # create output directory to backup moved data
            if not os.path.exists(args.backup):
                os.mkdir(args.backup)
            bk_dir = os.path.join(args.backup, 'BioMotionLab_NTroje')
            if not os.path.exists(bk_dir):
                os.mkdir(bk_dir)
            subj_names = sorted([f for f in os.listdir(dataset_path) if f[0] != '.'])
            subj_paths = sorted([os.path.join(dataset_path, f) for f in subj_names])
            bk_subj_paths = sorted([os.path.join(bk_dir, f) for f in subj_names])
            # print(subj_paths)
            for subj_dir, bk_subj_dir in zip(subj_paths, bk_subj_paths):
                motion_paths = sorted(glob.glob(subj_dir + '/*.npz'))
                # print(motion_paths)
                for motion_file in motion_paths:
                    motion_name = motion_file.split('/')[-1]
                    motion_type = motion_name.split('_')[1]
                    # print(motion_type)
                    if motion_type == 'treadmill' or motion_type == 'normal':
                        if not os.path.exists(bk_subj_dir):
                            os.mkdir(bk_subj_dir)
                        bk_path = os.path.join(bk_subj_dir, motion_name)
                        # print(bk_path)
                        shutil.move(motion_file, bk_path)


    if MPI_HDM05:
        # remove ice skating clips from MPI_HDM05
        # dg/HDM_dg_07-01* is inline skating
        dataset_path = os.path.join(args.data, 'MPI_HDM05')
        if not os.path.exists(dataset_path):
            print('Could not find MPI_HDM05 data, not filtering out inline skating sequences...')
        else:
            # create output directory to backup moved data
            if not os.path.exists(args.backup):
                os.mkdir(args.backup)
            bk_dir = os.path.join(args.backup, 'MPI_HDM05')
            if not os.path.exists(bk_dir):
                os.mkdir(bk_dir)
            subj_path = os.path.join(dataset_path, 'dg')
            if not os.path.exists(subj_path):
                print('Could not find problematic subject in MPI_HDM05: dg')
            else:
                skating_clips = sorted(glob.glob(subj_path + '/HDM_dg_07-01*'))
                # print(skating_clips)
                # print(len(skating_clips))
                bk_dir = os.path.join(bk_dir, 'dg')
                if not os.path.exists(bk_dir):
                    os.mkdir(bk_dir)

                for clip in skating_clips:
                    bk_path = os.path.join(bk_dir, clip.split('/')[-1])
                    # print(bk_path)
                    shutil.move(clip, bk_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='Root dir of processed AMASS data')
    parser.add_argument('--backup', type=str, required=True, help='Root directory to save removed data to.')

    config = parser.parse_known_args()
    config = config[0]

    main(config)