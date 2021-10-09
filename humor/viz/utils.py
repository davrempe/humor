
import time, os, shutil
import subprocess
import glob

import numpy as np
import trimesh
import torch
from PIL import Image, ImageDraw, ImageFont

from utils.torch import copy2cpu as c2c

smpl_connections = [[11, 8], [8, 5], [5, 2], [2, 0], [10, 7], [7, 4], [4, 1], [1, 0], 
                [0, 3], [3, 6], [6, 9], [9, 12], [12, 15], [12, 13], [13, 16], [16, 18], 
                [18, 20], [12, 14], [14, 17], [17, 19], [19, 21]]

imapper_connections = [[0, 1], [1, 2], [5, 4], [4, 3], [2, 6], [3, 6], [6, 7], [7, 8], [8, 9],
                       [7, 12], [12, 11], [11, 10], [7, 13], [13, 14], [14, 15]]

comp_connections = [[0, 1], [2, 3], [1, 4], [2, 4], [4, 5], [5, 8], [8, 7], [7, 6], [5, 9], [9, 10], [10, 11]]

colors = {
    'pink': [.7, .7, .9],
    'purple': [.9, .7, .7],
    'cyan': [.7, .75, .5],
    'red': [1.0, 0.0, 0.0],

    'green': [.0, 1., .0],
    'yellow': [1., 1., 0],
    'brown': [.5, .7, .7],
    'blue': [.0, .0, 1.],

    'offwhite': [.8, .9, .9],
    'white': [1., 1., 1.],
    'orange': [.5, .65, .9],

    'grey': [.7, .7, .7],
    'black': np.zeros(3),
    'white': np.ones(3),

    'yellowg': [0.83, 1, 0],
}

def create_video(img_path, out_path, fps):
    '''
    Creates a video from the frame format in the given directory and saves to out_path.
    '''
    command = ['ffmpeg', '-y', '-r', str(fps), '-i', img_path, \
                    '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', out_path]
    subprocess.run(command)

def create_gif(img_path, out_path, fps):
    '''
    Creates a gif (and video) from the frame format in the given directory and saves to out_path.
    '''
    vid_path = out_path[:-3] + 'mp4'
    create_video(img_path, vid_path, fps)
    subprocess.run(['ffmpeg', '-y', '-i', vid_path, \
                    '-pix_fmt', 'rgb8', out_path])

def create_comparison_images(img1_dir, img2_dir, out_dir, text1=None, text2=None):
    '''
    Given two direcdtories containing (png) frames of a video, combines them into one large frame and
    saves new side-by-side images to the given directory.
    '''
    img1_frames = sorted(glob.glob(os.path.join(img1_dir, '*.png')))
    img2_frames = sorted(glob.glob(os.path.join(img2_dir, '*.png')))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for img1_path, img2_path in zip(img1_frames, img2_frames):
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if text1 is not None:
            d = ImageDraw.Draw(img1)
            font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
            d.text((10, 10), text1, fill=(0,0,0), font=font)
        if text2 is not None:
            d = ImageDraw.Draw(img2)
            font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
            d.text((10, 10), text2, fill=(0,0,0), font=font)

        dst = Image.new('RGB', (img1.width + img2.width, img1.height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))

        dst.save(os.path.join(out_dir, img1_path.split('/')[-1]))

def create_multi_comparison_images(img_dirs, out_dir, texts=None, extn='.png'):
    '''
    Given list of direcdtories containing (png) frames of a video, combines them into one large frame and
    saves new side-by-side images to the given directory.
    '''
    img_frame_list = []
    for img_dir in img_dirs:
        img_frame_list.append(sorted(glob.glob(os.path.join(img_dir, '*.' + extn))))

    use_text = texts is not None and len(texts) == len(img_dirs)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for img_path_tuple in zip(*img_frame_list):
        img_list = []
        width_list = []
        for im_idx, cur_img_path in enumerate(img_path_tuple):
            cur_img = Image.open(cur_img_path)
            if use_text:
                d = ImageDraw.Draw(cur_img)
                font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 12)
                d.text((10, 10), texts[im_idx], fill=(0,0,0), font=font)
            img_list.append(cur_img)
            width_list.append(cur_img.width)

        dst = Image.new('RGB', (sum(width_list), img_list[0].height))
        for im_idx, cur_img in enumerate(img_list):
            if im_idx == 0:
                dst.paste(cur_img, (0, 0))
            else:
                dst.paste(cur_img, (sum(width_list[:im_idx]), 0))

        dst.save(os.path.join(out_dir, img_path_tuple[0].split('/')[-1]))

def viz_smpl_seq(body, imw=1080, imh=1080, fps=30, contacts=None,
                render_body=True, render_joints=False, render_skeleton=False, render_ground=True, ground_plane=None,
                use_offscreen=False, out_path=None, wireframe=False, RGBA=False,
                joints_seq=None, joints_vel=None, follow_camera=False, vtx_list=None, points_seq=None, points_vel=None,
                static_meshes=None, camera_intrinsics=None, img_seq=None, point_rad=0.015,
                skel_connections=smpl_connections, img_extn='png', ground_alpha=1.0, body_alpha=None, mask_seq=None,
                cam_offset=[0.0, 4.0, 1.25], ground_color0=[0.8, 0.9, 0.9], ground_color1=[0.6, 0.7, 0.7],
                skel_color=[0.0, 0.0, 1.0],
                joint_rad=0.015,
                point_color=[0.0, 1.0, 0.0],
                joint_color=[0.0, 1.0, 0.0],
                contact_color=[1.0, 0.0, 0.0],
                render_bodies_static=None,
                render_points_static=None,
                cam_rot=None):
    '''
    Visualizes the body model output of a smpl sequence.
    - body : body model output from SMPL forward pass (where the sequence is the batch)
    - joints_seq : list of torch/numy tensors/arrays
    - points_seq : list of torch/numpy tensors
    - camera_intrinsics : (fx, fy, cx, cy)
    - ground_plane : [a, b, c, d]
    - render_bodies_static is an integer, if given renders all bodies at once but only every x steps
    '''

    if contacts is not None and torch.is_tensor(contacts):
        contacts = c2c(contacts)

    if render_body or vtx_list is not None:
        start_t = time.time()
        nv = body.v.size(1)
        vertex_colors = np.tile(colors['grey'], (nv, 1))
        if body_alpha is not None:
            vtx_alpha = np.ones((vertex_colors.shape[0], 1))*body_alpha
            vertex_colors = np.concatenate([vertex_colors, vtx_alpha], axis=1)
        faces = c2c(body.f)
        body_mesh_seq = [trimesh.Trimesh(vertices=c2c(body.v[i]), faces=faces, vertex_colors=vertex_colors, process=False) for i in range(body.v.size(0))]

    if render_joints and joints_seq is None:
        start_t = time.time()
        # only body joints
        joints_seq = [c2c(body.Jtr[i, :22]) for i in range(body.Jtr.size(0))]
    elif render_joints and torch.is_tensor(joints_seq[0]):
        joints_seq = [c2c(joint_frame) for joint_frame in joints_seq]

    if joints_vel is not None and torch.is_tensor(joints_vel[0]):
        joints_vel = [c2c(joint_frame) for joint_frame in joints_vel]
    if points_vel is not None and torch.is_tensor(points_vel[0]):
        points_vel = [c2c(joint_frame) for joint_frame in points_vel]

    mv = MeshViewer(width=imw, height=imh,
                    use_offscreen=use_offscreen, 
                    follow_camera=follow_camera,
                    camera_intrinsics=camera_intrinsics,
                    img_extn=img_extn,
                    default_cam_offset=cam_offset,
                    default_cam_rot=cam_rot)
    if render_body and render_bodies_static is None:
        mv.add_mesh_seq(body_mesh_seq)
    elif render_body and render_bodies_static is not None:
        mv.add_static_meshes([body_mesh_seq[i] for i in range(len(body_mesh_seq)) if i % render_bodies_static == 0])
    if render_joints and render_skeleton:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, 
                         connections=skel_connections, connect_color=skel_color, vel=joints_vel,
                         contact_color=contact_color, render_static=render_points_static)
    elif render_joints:
        mv.add_point_seq(joints_seq, color=joint_color, radius=joint_rad, contact_seq=contacts, vel=joints_vel, contact_color=contact_color,
                            render_static=render_points_static)

    if vtx_list is not None:
        mv.add_smpl_vtx_list_seq(body_mesh_seq, vtx_list, color=[0.0, 0.0, 1.0], radius=0.015)

    if points_seq is not None:
        if torch.is_tensor(points_seq[0]):
            points_seq = [c2c(point_frame) for point_frame in points_seq]
        mv.add_point_seq(points_seq, color=point_color, radius=point_rad, vel=points_vel, render_static=render_points_static)

    if static_meshes is not None:
        mv.set_static_meshes(static_meshes)

    if img_seq is not None:
        mv.set_img_seq(img_seq)

    if mask_seq is not None:
        mv.set_mask_seq(mask_seq)

    if render_ground:
        xyz_orig = None
        if ground_plane is not None:
            if render_body:
                xyz_orig = body_mesh_seq[0].vertices[0, :]
            elif render_joints:
                xyz_orig = joints_seq[0][0, :]
            elif points_seq is not None:
                xyz_orig = points_seq[0][0, :]

        mv.add_ground(ground_plane=ground_plane, xyz_orig=xyz_orig, color0=ground_color0, color1=ground_color1, alpha=ground_alpha)

    mv.set_render_settings(out_path=out_path, wireframe=wireframe, RGBA=RGBA,
                            single_frame=(render_points_static is not None or render_bodies_static is not None)) # only does anything for offscreen rendering
    try:
        mv.animate(fps=fps)
    except RuntimeError as err:
        print('Could not render properly with the error: %s' % (str(err)))

    del mv

def viz_results(body_pred, body_gt, fps, viz_out_dir=None, base_name=None, contacts=None, pred_joints=None, gt_joints=None,
                pred_verts=None, gt_verts=None, render_body=True, cleanup=True, pred_contacts=None, gt_contacts=None,
                wireframe=False, RGBA=False, camera_intrinsics=None, imw=1080, imh=1080, img_seq=None,
                render_ground=True, point_rad=0.015, ground_plane=None, render_pred_body=None, render_gt_body=None,
                skel_connections=smpl_connections, ground_alpha=1.0, body_alpha=None, point_color=[0.0, 1.0, 0.0],
                cam_offset=[0.0, 4.0, 1.25]):
    use_offscreen = False
    pred_out_path = gt_out_path = comparison_out_path = None
    if viz_out_dir is not None:
        if base_name is None:
            print('Must give base name to save visualized output')
            return
        use_offscreen = True
        if not os.path.exists(viz_out_dir):
            os.mkdir(viz_out_dir)

        base_out_path = os.path.join(viz_out_dir, base_name)
        pred_out_path = base_out_path + '_pred'
        gt_out_path = base_out_path + '_gt'
        comparison_out_path = base_out_path + '_compare'

    if pred_contacts is None:
        pred_contacts = contacts
    if gt_contacts is None:
        gt_contacts = contacts

    if render_pred_body is not None or render_gt_body is not None:
        if render_pred_body is None:
            render_pred_body = render_body
        if render_gt_body is None:
            render_gt_body = render_body
    else:
        render_pred_body = render_body
        render_gt_body = render_body

    # determine whether to have a following camera or not
    follow_camera = torch.max(torch.abs(body_pred.Jtr[:, :22, :2])) > 2.0
    if follow_camera:
        print('Using follow camera...')
    print('Visualizing PREDICTED sequence...')
    viz_smpl_seq(body_pred, 
                 imw=imw,
                 imh=imh,
                 fps=fps,
                 contacts=pred_contacts,
                 render_body=render_pred_body,
                 render_joints=(pred_joints is not None),
                 render_skeleton=(not render_pred_body),
                 render_ground=render_ground,
                 ground_plane=ground_plane,
                 ground_alpha=ground_alpha,
                 body_alpha=body_alpha,
                 joints_seq=pred_joints,
                 points_seq=pred_verts,
                 use_offscreen=use_offscreen,
                 out_path=pred_out_path,
                 wireframe=wireframe,
                 RGBA=RGBA,
                 point_rad=point_rad,
                 camera_intrinsics=camera_intrinsics,
                 follow_camera=follow_camera,
                 img_seq=img_seq,
                 skel_connections=skel_connections,
                 point_color=point_color,
                 cam_offset=cam_offset)
    print('Visualizing GROUND TRUTH sequence...')
    viz_smpl_seq(body_gt, 
                 imw=imw,
                 imh=imh,
                 fps=fps,
                 contacts=gt_contacts,
                 render_body=render_gt_body,
                 render_joints=(gt_joints is not None),
                 render_skeleton=(not render_gt_body),
                 render_ground=render_ground,
                 ground_plane=ground_plane,
                 ground_alpha=ground_alpha,
                 body_alpha=body_alpha,
                 joints_seq=gt_joints,
                 points_seq=gt_verts,
                 use_offscreen=use_offscreen,
                 out_path=gt_out_path,
                 wireframe=wireframe,
                 RGBA=RGBA,
                 point_rad=point_rad,
                 camera_intrinsics=camera_intrinsics,
                 follow_camera=follow_camera,
                 img_seq=img_seq,
                 skel_connections=skel_connections,
                 point_color=point_color,
                 cam_offset=cam_offset)

    if use_offscreen:
        # create a video of each
        # create_video(os.path.join(pred_out_path + '/frame_%08d.png'), pred_out_path + '.mp4', fps)
        # create_video(os.path.join(gt_out_path + '/frame_%08d.png'), gt_out_path + '.mp4', fps)
        # # then for comparison
        # create_comparison_images(gt_out_path, pred_out_path, comparison_out_path, text1='GT', text2='Pred')
        # create_video(os.path.join(comparison_out_path + '/frame_%08d.png'), comparison_out_path + '.mp4', fps)    

        # or gif
        create_gif(os.path.join(pred_out_path + '/frame_%08d.png'), pred_out_path + '.gif', fps)
        create_gif(os.path.join(gt_out_path + '/frame_%08d.png'), gt_out_path + '.gif', fps)
        create_comparison_images(gt_out_path, pred_out_path, comparison_out_path, text1='GT', text2='Pred')
        create_gif(os.path.join(comparison_out_path + '/frame_%08d.png'), comparison_out_path + '.gif', fps)

        # cleanup
        if cleanup:
            shutil.rmtree(pred_out_path)
            shutil.rmtree(gt_out_path)
            shutil.rmtree(comparison_out_path)

# avoid cyclic dependency
from viz.mesh_viewer import MeshViewer
