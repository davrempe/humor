import os, sys, shutil, argparse, subprocess, time, json, glob
from multiprocessing import Pool

import os.path as osp
import torch, torchvision
from PIL import Image
from torchvision import transforms
import numpy as np

import cv2


def video_to_images(vid_file, img_folder=None, return_info=False, fps=30):
    '''
    From https://github.com/mkocabas/VIBE/blob/master/lib/utils/demo_utils.py

    fps will sample the video to this rate.
    '''
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', str(fps),
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder

def make_absolute(rel_paths):
    ''' Makes a list of relative paths absolute '''
    return [os.path.join(os.getcwd(), rel_path) for rel_path in rel_paths]

SKELETON = 'BODY_25'

def run_openpose(openpose_path, img_dir, out_dir, video_out=None, img_out=None):
    '''
    Runs OpenPose for 2D joint detection on the images in img_dir.
    '''
    # make all paths absolute to call OP
    openpose_path = make_absolute([openpose_path])[0]
    img_dir = make_absolute([img_dir])[0]
    out_dir = make_absolute([out_dir])[0]
    if video_out is not None:
        video_out = make_absolute([video_out])[0]
    if img_out is not None:
        img_out = make_absolute([img_out])[0]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # run open pose
    # must change to openpose dir path to run properly
    og_cwd = os.getcwd()
    os.chdir(openpose_path)

    # then run openpose
    run_cmds = ['./build/examples/openpose/openpose.bin', \
                '--image_dir', img_dir, '--write_json', out_dir, \
                '--display', '0', '--model_pose', SKELETON, '--number_people_max', '1']
    if video_out is not None:
        run_cmds +=  ['--write_video', video_out, '--write_video_fps', '30']
    if img_out is not None:
        run_cmds += ['--write_images', img_out]
    if not (video_out is not None or img_out is not None):
        run_cmds += ['--render_pose', '0']
    print(run_cmds)
    subprocess.run(run_cmds)

    os.chdir(og_cwd) # change back to resume


def run_deeplab_v3(img_dir, img_shape, out_dir, batch_size=16, img_extn='png'):
    '''
    Runs DeepLabv3 to get a person segmentation mask on each img in img_dir.
    
    - img_shape : (H x W)
    '''
    print('Running DeepLabv3 to compute person mask...')
    H, W = img_shape
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True).to(device)
    model.eval()
    preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    img_path = img_dir
    all_img_paths = sorted(glob.glob(os.path.join(img_path + '/*.'  + img_extn)))
    img_names = ['.'.join(f.split('/')[-1].split('.')[:-1]) for f in all_img_paths]
    out_path = out_dir
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_mask_paths = [os.path.join(out_path, f + '.png') for f in img_names]
    # print(all_mask_paths)

    num_imgs = len(img_names)
    num_batches = (num_imgs / batch_size) + 1
    sidx = 0
    eidx = min(num_imgs, batch_size)
    cnt = 1
    while sidx < num_imgs:
        # print(sidx)
        # print(eidx)
        # batch
        print('Batch %d / %d' % (cnt, num_batches))
        img_path_batch = all_img_paths[sidx:eidx]
        mask_path_batch = all_mask_paths[sidx:eidx]
        B = len(img_path_batch)
        img_batch = torch.zeros((B, 3, H, W))
        for bidx, cur_img_path in enumerate(img_path_batch):
            input_image = Image.open(cur_img_path)
            input_tensor = preprocess(input_image)
            img_batch[bidx] = input_tensor
        img_batch = img_batch.to(device)
        # print(img_batch.size())

        # eval and save
        with torch.no_grad():
            output = model(img_batch)['out']
        seg = torch.logical_not(output.argmax(1) == 15).to(torch.float) # the max probability is the person class
        seg = seg.cpu().numpy()
        for bidx in range(B):
            person_mask = (seg[bidx]*255.0).astype(np.uint8)
            out_img = Image.fromarray(person_mask)
            out_img.save(mask_path_batch[bidx])


        # # create a color pallette, selecting a color for each class
        # palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        # colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
        # colors = (colors % 255).numpy().astype("uint8")
        # # plot the semantic segmentation predictions of 21 classes in each color
        # r = Image.fromarray(seg[0].byte().cpu().numpy()).resize(input_image.size)
        # r.putpalette(colors)
        # import matplotlib.pyplot as plt
        # plt.imshow(r)
        # plt.show()

        sidx = eidx
        eidx = min(num_imgs, sidx + batch_size)
        cnt += 1