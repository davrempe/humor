# HuMoR: 3D Human Motion Model for Robust Pose Estimation (ICCV 2021)
This is the official implementation for the ICCV 2021 paper. For more information, see the [project webpage](https://geometry.stanford.edu/projects/humor/).

![HuMoR Teaser](humor.png)

## Environment Setup
> Note: This code was developed on Ubuntu 16.04/18.04 with Python 3.7, CUDA 10.1 and PyTorch 1.6.0. Later versions should work, but have not been tested.

Create and activate a virtual environment to work in, e.g. using Conda:
```
conda create -n humor_env python=3.7
conda activate humor_env
```

Install CUDA and PyTorch 1.6. For CUDA 10.1, this would look like:
```
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

You must also have _ffmpeg_ installed on your system to save visualizations.

## Downloads & External Dependencies
This codebase relies on various external downloads in order to run for certain modes of operation. Here we briefly overview each and what they are used for. Detailed setup instructions are linked in other READMEs.

### Body Model and Pose Prior 
Detailed instructions to install SMPL+H and VPoser are in [this documentation](./body_models/).

* [SMPL+H](https://mano.is.tue.mpg.de/) is used for the pose/shape body model. Downloading this model is necessary for **all uses** of this codebase.
* [VPoser](https://github.com/nghorbani/human_body_prior) is used as a pose prior only during the initialization phase of fitting, so it's only needed if you are using the test-time optimization functionality of this codebase.

### Datasets
Detailed instructions to install, configure, and process each dataset are in [this documentation](./data/).

* [AMASS](https://amass.is.tue.mpg.de/) motion capture data is used to train and evaluate (_e.g._ randomly sample) the HuMoR motion model and for fitting to 3D data like noisy joints and partial keypoints.
* [i3DB](https://github.com/amonszpart/iMapper) contains RGB videos with heavy occlusions and is only used in the paper to evaluate test-time fitting to 2D joints. 
* [PROX](https://prox.is.tue.mpg.de/) contains RGB-D videos and is only used in the paper to evaluate test-time fitting to 2D joints and 3D point clouds.

### Pretrained Models
Pretrained model checkpoints are available for HuMoR, HuMoR-Qual, and the initial state Gaussian mixture. To download (~215 MB), from the repo root run `bash get_ckpt.sh`.

### OpenPose
[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is used to detect 2D joints for fitting to arbitrary RGB videos. If you will be running test-time optimization on the demo video or your own videos, you must install OpenPose (unless you pass in pre-computed OpenPose results using `--op-keypts`). To clone and build, please follow the [OpenPose README](https://github.com/CMU-Perceptual-Computing-Lab/openpose) in their repo.

Optimization in [run_fitting.py](./humor/fitting/run_fitting.py) assumes OpenPose is installed at `./external/openpose` by default - if you install elsewhere, please pass in the location using the `--openpose` flag. 

## Fitting to RGB Videos (Test-Time Optimization)
To run motion/shape estimation on an arbitrary RGB video, you must have SMPL+H, VPoser, OpenPose, and a pretrained HuMoR model as detailed above. We have included a demo video in this repo along with a few example configurations to get started.

> Note: if running on your own video, make sure the camera is not moving and the person is not interacting with uneven terrain in the scene (we assume a single ground plane). Also, only one person will be reconstructed.

To run the optimization on the demo video use:
```
python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_split.cfg
```

This configuration optimizes over the entire video (~3 sec) at once (i.e. over all frames). **If your video is longer than 2-3 sec**, it is recommended to instead use the settings in `./configs/fit_rgb_demo_use_split.cfg` which adds the `--rgb-seq-len`, `--rgb-overlap-len`, and `--rgb-overlap-consist-weight` arguments. Using this configuration, the input video is split into multiple overlapping sub-sequences and optimized in a batched fashion (with consistency losses between sub-sequences). This increases efficiency, and lessens the need to tune parameters based on video length. Note the larger the batch size, the better the results will be.

If known, it's **highly recommended to pass in camera intrinsics** using the `--rgb-intrinsics` flag. See `./configs/intrinsics_default.json` for an example of what this looks like. If intrinsics are _not_ given, [default focal lengths](./humor/fitting/fitting_utils.py#L19) are used.

Finally, this demo does _not_ use [PlaneRCNN](https://github.com/NVlabs/planercnn) to initialize the ground as described in the paper. Instead, it roughly initializes the ground at `y = 0.5` (with camera up-axis `-y`). We found this to be sufficient and often better than using PlaneRCNN. If you want to use PlaneRCNN instead, set up a separate environment, follow their install instructions, then use the following command to run their method where `example_image_dir` contains a single frame from your video and the camera parameters: `python evaluate.py --methods=f --suffix=warping_refine --dataset=inference --customDataFolder=example_image_dir`. The results directory can be passed into our optimization using the `--rgb-planercnn-res` flag.

> Note: if you want to use your own OpenPose detections rather than having the fitting script run OpenPose, pass in the directory containing the json files using `--op-keypts`. The [expected format](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md#json-output-format) of these json files is that written using the `--write_json` flag when running OpenPose with the `BODY_25` skeleton and unnormalized 2D keypoints (see [here](https://github.com/davrempe/humor/blob/main/humor/utils/video.py#L70) for the exact OpenPose command we use). We also assume these OpenPose detections are at 30 fps since this is the rate of the HuMoR model. Note that only the `pose_keypoints_2d` (body joints) of the first detected person in each json file is used for fitting.

### Visualizing RGB Results
The optimization is performed in 3 stages, with stages 1 & 2 being initialization using a pose prior and smoothing (i.e. the _VPoser-t_ baseline) and stage 3 being the full optimization with the HuMoR motion prior. So for the demo, the final output for the full sequence will be saved in `./out/rgb_demo_no_split/results_out/final_results/stage3_results.npz`. To visualize results from the fitting use something like:
```
python humor/fitting/viz_fitting_rgb.py  --results ./out/rgb_demo_no_split/results_out --out ./out/rgb_demo_no_split/viz_out --viz-prior-frame
```

By default, this will visualize the final full video result along with each sub-sequence separately (if applicable). Please use `--help` to see the many additional visualization options. This code is also useful to see how to load in and use the results for other tasks, if desired.

## Fitting on Specific Datasets
Next, we detail how to run and evaluate the test-time optimization on the various datasets presented in the paper. In all these examples, the default batch size is quite small to accomodate smaller GPUs, but it should be increased depending on your system.

### AMASS 3D Data
There are multiple settings possible for fitting to 3D data (e.g. noisy joints, partial keypoints, etc...), which can be specified using configuration flags. For example, to fit to partial upper-body 3D keypoints sampled from AMASS data, run:
```
python humor/fitting/run_fitting.py @./configs/fit_amass_keypts.cfg
```

Optimization results can be visualized using
```
python humor/fitting/eval_fitting_3d.py --results ./out/amass_verts_upper_fitting/results_out --out ./out/amass_verts_upper_fitting/eval_out  --qual --viz-stages --viz-observation
```
and evaluation metrics computed with
```
python humor/fitting/eval_fitting_3d.py --results ./out/amass_verts_upper_fitting/results_out --out ./out/amass_verts_upper_fitting/eval_out  --quant --quant-stages
```

The most relevant quantitative results will be written to `eval_out/eval_quant/compare_mean.csv`.

### i3DB RGB Data
The i3DB dataset contains RGB videos with many occlusions along with annotated 3D joints for evaluation. To run test-time optimization on the full dataset, use:
```
python humor/fitting/run_fitting.py @./configs/fit_imapper.cfg
```

Results can be visualized using the same script as in the demo:
```
python humor/fitting/viz_fitting_rgb.py  --results ./out/imapper_fitting/results_out --out ./out/imapper_fitting/viz_out --viz-prior-frame
```

Quantitative evaluation (comparing to results after each optimization stage) can be run with:
```
python humor/fitting/eval_fitting_2d.py --results ./out/imapper_fitting/results_out --dataset iMapper --imapper-floors ./data/iMapper/i3DB/floors --out ./out/imapper_fitting/eval_out --quant --quant-stages
```

The final quantitative results will be written to `eval_out/eval_quant/compare_mean.csv`.

### PROX RGB/RGB-D Data
PROX contains RGB-D data so affords fitting to just 2D joints and 2D joints + 3D point cloud. The commands for running each of these are quite similar, just using different configuration files. For running on the full RGB-D data, use:
```
python humor/fitting/run_fitting.py @./configs/fit_proxd.cfg
```

Visualization must add the `--flip-img` flag to align with the original PROX videos:
```
python humor/fitting/viz_fitting_rgb.py  --results ./out/proxd_fitting/results_out --out ./out/proxd_fitting/viz_out --viz-prior-frame --flip-img
```

Quantitative evalution (of plausibility metrics) for full RGB-D data uses
```
python humor/fitting/eval_fitting_2d.py --results ./out/proxd_fitting/results_out --dataset PROXD --prox-floors ./data/prox/qualitative/floors --out ./out/proxd_fitting/eval_out --quant --quant-stages
```

and for just RGB data is slightly different:
```
python humor/fitting/eval_fitting_2d.py --results ./out/prox_fitting/results_out --dataset PROX --prox-floors ./data/prox/qualitative/floors --out ./out/prox_fitting/eval_out --quant --quant-stages
```

## Training & Testing Motion Model
There are two versions of our model: HuMoR and HuMoR-Qual. HuMoR is the main model presented in the paper and is best suited for test-time optimization. HuMoR-Qual is a slight variation on HuMoR that gives more stable and qualitatively superior results for random motion generation (see the paper for details).

Below we describe how to train and test HuMoR, but the exact same commands are used for HuMoR-Qual with a different configuration file at each step (see [all provided configs](./configs)).

### Training HuMoR
To train HuMoR from scratch, make sure you have the processed version of the AMASS dataset at `./data/amass_processed` and run:
```
python humor/train/train_humor.py @./configs/train_humor.cfg
```
The default batch size is meant for a 16 GB GPU.

### Testing HuMoR
After training HuMoR or downloading the pretrained checkpoints, we can evaluate the model in multiple ways

To compute single-step losses (the exact same as during training) over the entire test set run:
```
python humor/test/test_humor.py @./configs/test_humor.cfg
```

To randomly sample a motion sequence and save a video visualization, run:
```
python humor/test/test_humor.py @./configs/test_humor_sampling.cfg
```

If you'd rather visualize the sampling results in an interactive viewer, use:
```
python humor/test/test_humor.py @./configs/test_humor_sampling_debug.cfg
```

Try adding `--viz-pred-joints`, `--viz-smpl-joints`, or `--viz-contacts` to the end of the command to visualize more outputs, or increasing the value of `--eval-num-samples` to sample the model multiple times from the same initial state. `--help` can always be used to see all flags and their descriptions.

To reconstruct random sequences from AMASS (i.e. encode then decode them), use:
```
python humor/test/test_humor.py @./configs/test_humor_recon.cfg
```

### Training Initial State GMM
Test-time optimization also uses a Gaussian mixture model (GMM) prior over the initial state of the sequence. The pretrained model can be downloaded above, but if you wish to train from scratch, run:
```
python humor/train/train_state_prior.py --data ./data/amass_processed --out ./out/init_state_prior_gmm --gmm-comps 12
```

## Citation
If you found this code or paper useful, please consider citing:
```
@inproceedings{rempe2021humor,
    author={Rempe, Davis and Birdal, Tolga and Hertzmann, Aaron and Yang, Jimei and Sridhar, Srinath and Guibas, Leonidas J.},
    title={HuMoR: 3D Human Motion Model for Robust Pose Estimation},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2021}
}
```

## Questions?
If you run into any problems or have questions, please create an issue or contact Davis (first author) via email.
