import os
import random
import imageio.v2 as imageio
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas
from tqdm import tqdm

# Import data readers / generators
from dataset.my_mesh import MyMesh

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render
from render import load_glb

RADIUS = 3.0

def z_buffering(rasterized_outputs, antialiased_images):
    rasterized_outputs = torch.cat(rasterized_outputs, dim=0)  # Shape: [num_components, height, width, 4] --> 3rd channel gives depth value
    antialiased_images = torch.cat(antialiased_images, dim=0) # Shape: [num_components, height, width, num_channels]
    _, height, width, num_channels = antialiased_images.shape

    # Initialize the z-buffer and final color buffer
    z_buffer = torch.ones((height, width)) * float('inf')
    z_buffer = z_buffer.cuda()
    final_image = torch.zeros((height, width, num_channels))
    final_image = antialiased_images[0,:,:,:].squeeze()
    final_image = final_image.cuda()

    # Combine the components using z-buffering
    for index in range(len(rasterized_outputs)):
        depth = rasterized_outputs[index, :,:,2]
        # torch.set_printoptions(profile="full")
        # print(depth)
        depth = depth.cuda()
        color = antialiased_images[index,:,:,:]
        color = color.cuda()
        # Find the pixels where the current component is closer than the existing z-buffer
        closer_pixels = depth < z_buffer
        closer_pixels = closer_pixels.cuda()
        z_buffer[closer_pixels] = depth[closer_pixels]
        final_image[closer_pixels] = color[closer_pixels]

    return final_image.detach().cpu().numpy()

def load_materials(length):
    materials_dir = "materials/"
    num_materials_to_select = length

    # Get a list of all material directories
    all_material_dirs = os.listdir(materials_dir)

    # Randomly select N materials from the list
    selected_material_dirs = random.sample(all_material_dirs, num_materials_to_select)

    # Initialize an empty list to store the loaded materials
    loaded_materials = []

    # Load each selected material
    for material_dir in selected_material_dirs:
        material_path = os.path.join(materials_dir, material_dir)

        number = random.randint(0, 9)
        base_color_file = os.path.join(material_path, f"{number}/", "basecolor", 'basecolor.png')
        roughness_file = os.path.join(material_path, f"{number}/", "roughness", 'roughness.png')
        metallic_file = os.path.join(material_path, f"{number}/", "metallic", 'metallic.png')
        normal_file = os.path.join(material_path, f"{number}/", "normal", 'normal.png')

        # Load each material PNG file as numpy array
        base_color = imageio.imread(base_color_file) / 255.0  # Normalize to [0, 1]
        roughness = imageio.imread(roughness_file) / 255.0
        metallic = imageio.imread(metallic_file) / 255.0
        normal = imageio.imread(normal_file) / 255.0
        print('base color')
        print(base_color.shape)
        # Create the dictionary for the material
        material_dict = {
            'name': material_dir,  # Use the material directory name as the material name
            'base_color': base_color.astype(np.float32),
            'roughness': roughness.astype(np.float32),
            'metallic': metallic.astype(np.float32),
            'normal': normal.astype(np.float32)
        }

        # Append the material dictionary to the list
        loaded_materials.append(material_dict)
    return loaded_materials


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    
    FLAGS = parser.parse_args()
    # FLAGS.config = 'configs/3dcompat_chair.json'
    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = 'out/' + FLAGS.out_dir

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    outs = []
    rasterized_outputs = []
    antialised_images = []
    depths = []
    
    if os.path.splitext(FLAGS.ref_mesh)[1] == '.glb':
        length, model_components, _  = load_glb.load_glb(FLAGS.ref_mesh)
        materials = load_materials(length)
        for ind in range(length):
            ref_mesh = mesh.load_mesh(FLAGS.ref_mesh, ind, length, model_components, materials, FLAGS.mtl_override)
            my_mesh = MyMesh(ref_mesh, glctx, RADIUS, FLAGS)
            out = my_mesh.get_output()
            outs.append(out)
            x = out['rast'].squeeze()
            depth = x[:,:,2]
            depth = depth.detach().cpu().numpy()
            print(depth.shape)
            depths.append(depth)
            rasterized_outputs.append(out['rast'].squeeze().detach().cpu().numpy())
            antialised_images.append(out['img'].squeeze().detach().cpu().numpy())
    
    # print('performing z-buffering')
    # img = z_buffering(rasterized_outputs, antialised_images)
    
    # print('saving the final image obtained')
    # util.save_image('images/output_final.png', img)
    
    canvas_ht = antialised_images[0].shape[0]
    canvas_wd = antialised_images[0].shape[1]
    canvas_channels = antialised_images[0].shape[2]
    canvas = np.zeros((canvas_ht, canvas_wd, canvas_channels), dtype=np.float32)
    canvas_depth = np.full((canvas_ht, canvas_wd), np.inf, dtype=np.float32)
    
    for image, depth in zip(antialised_images, depths):
        # Iterate over each pixel in the image
        for i in tqdm(range(canvas_ht)):
            for j in range(canvas_wd):
                # Retrieve the depth value for the current pixel
                pixel_depth = depth[i, j]
                # Check if the current pixel depth is less than the depth in the canvas
                if pixel_depth != 0 and (pixel_depth < canvas_depth[i, j]):
                    # print(pixel_depth, canvas_depth[i,j])
                    # Update the corresponding pixel in the canvas with the image pixel value
                    canvas[i, j, :] = image[i, j, :]
                    canvas_depth[i,j] = pixel_depth
    util.save_image(f'combined_image.png', canvas)
    # for ind in range(6):
    #     img = antialised_images[ind].squeeze().detach().cpu().numpy()
    #     util.save_image(f'images/out_{ind}.png', img)
