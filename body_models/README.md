Both [SMPL+H](https://mano.is.tue.mpg.de/) and [VPoser](https://github.com/nghorbani/human_body_prior) should be installed to this directory. Detailed instructions are below. After installation, this directory should contain a `smplh` directory and a `vposer_v1_0` directory.

## SMPL+H
To install the body model:
* Create an account on the [project page](https://mano.is.tue.mpg.de/)
* Go to the `Downloads` page and download the "Extended SMPL+H model (used in AMASS)". Place the downloaded `smplh.tar.xz` in this directory.
* Extract downloaded model to new directory `mkdir smplh && tar -xf smplh.tar.xz -C smplh`. The model will be read in from here automatically when running this codebase.

Note if you decide to install the body model somewhere else, please update `SMPLH_PATH` in [this file](../humor/body_model/utils.py).

## VPoser
To install the pose prior:
* Create an account on the [project page](https://smpl-x.is.tue.mpg.de/index.html)
* Go to the `Download` page and under "VPoser: Variational Human Pose Prior" click on "Download VPoser v1.0 - CVPR'19" (note it's important to download v1.0 and **not** v2.0 which is not supported and will not work)
* Copy the zip file to this directory, and unzip with `unzip vposer_v1_0.zip`

If you're left with a directory called `vposer_v1_0` in the current directory, then it's been successfully installed. The `--vposer` argument in `run_fitting.py` by default points to this directory.