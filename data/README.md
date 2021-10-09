By default, all datasets described in this README are expected to be placed in this directory. If you decide to install them elsewhere, you will need to modify various input arguments within the config files.

## AMASS
First you must obtain the raw AMASS dataset:
* Create a directory to place raw data in: `mkdir amass_raw`
* Create an account on the [project page](https://amass.is.tue.mpg.de/)
* Go to the `Downloads` page and download the SMPL+H Body Data for all datasets. Extract each dataset to its own directory within `amass_raw` (_e.g._ all CMU data should be at `data/amass_raw/CMU`).

Next, run the data processing which samples motions to 30 Hz, removes terrain interaction sequences, detects contacts, etc.., and saves the data into the format used by our codebase. From the root of this repo, run:
```
python humor/scripts/process_amass_data.py --amass-root ./data/amass_raw --out ./data/amass_processed --smplh-root ./body_models/smplh
```
By default this processes every sub-dataset in AMASS. If you only want to process a subset, e.g., CMU and HumanEva, pass in the flag `--datasets CMU HumanEva`.

A second script does some small extra cleanup to remove sequences we found to be problematic (e.g., walking/running on treadmill and ice skating which negatively affects learning the motion model):
```
python humor/scripts/cleanup_amass_data.py --data ./data/amass_processed --backup ./data/cleanup_bk
```
The `--backup` flag indicates a directory where the sequences that are removed will be saved in case you need them again later.

> Note: not all of the above processed data is actually used in training/testing HuMoR. To see the exact dataset splits used in the paper see [this script](../humor/datasets/amass_utils.py)

## i3DB
We have prepared a pre-processed version of the i3DB dataset, originally released with [iMapper](https://github.com/amonszpart/iMapper), that can be downloaded directly. To download, from this directory run:
```
bash get_i3db.sh
```

## PROX
We have prepared ground plane and 2D joint data that complement PROX in order to easily run our method on the dataset. The first step is to download the PROX dataset:
* First create the structure. From this directory run `mkdir prox && mkdir prox/qualitative`
* Create an account on the [project page](https://prox.is.tue.mpg.de/)
* Go to the `Download` page and download all files under "Qualitative PROX dataset" (note: `videos.zip` and `PROXD_videos.zip` are not required). Unzip these files to `prox/qualitative` that we created before.

You should now have the full PROX qualitative dataset with directory structure (if you downloaded all the optional files):
```
prox/qualitative
├── body_segments
├── calibration
├── cam2world
├── PROXD
├── PROXD_videos
├── recordings
├── scenes
└── sdf
```

To download the [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) 2D joint detections, ground truth floor planes, and [PlaneRCNN](https://github.com/NVlabs/planercnn) detections used in our paper, from this directory run `bash get_prox_extra.sh`. This will add `floors`, `keypoints`, and `planes` directories to the structure above.