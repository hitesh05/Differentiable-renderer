# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch

from . import util
from . import texture

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

######################################################################################
# Wrapper to make materials behave like a python dict, but register textures as 
# torch.nn.Module parameters.
######################################################################################
class Material(torch.nn.Module):
    def __init__(self, mat_dict):
        super(Material, self).__init__()
        self.mat_keys = set()
        for key in mat_dict.keys():
            self.mat_keys.add(key)
            self[key] = mat_dict[key]

    def __contains__(self, key):
        return hasattr(self, key)

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        self.mat_keys.add(key)
        setattr(self, key, val)

    def __delitem__(self, key):
        self.mat_keys.remove(key)
        delattr(self, key)

    def keys(self):
        return self.mat_keys


######################################################################################
# .mtl material format loading / storing
######################################################################################
@torch.no_grad()
def load_mtl(fn, mtrls=None, ind = 0, clear_ks=True):
    # ind = (ind+2)%5
    materials = []
    dt = {0: 'base_color', 1: 'roughness', 2: 'metallic', 3: 'normal'}
    material = Material({'name': 'Material'})
    material['bsdf'] = 'pbr'
    
    for i in range(4):
        x = mtrls[ind][dt[i]]
        if i == 0:
            material['map_kd'] = x
        elif i == 1:
            x1 = mtrls[ind][dt[i+1]]
            x2 = np.zeros_like(x1)
            x = np.stack((x2,x,x1), axis=-1)
            # x = np.pad(x, ((0, 0), (0, 0), (0, 1)), mode='constant')
            material['map_ks'] = x
        elif i ==3:
            material['map_normal'] = x
            print(material['map_normal'].shape)
            
        materials += [material]

    for ind, mat in enumerate(materials):
        if 'map_kd' in mat:
            mat['kd'] = texture.Texture2D(mat['map_kd'])
            # Convert Kd from sRGB to linear RGB
            mat['kd'] = texture.srgb_to_rgb(mat['kd'])    
            
        if 'map_ks' in mat:
            mat['ks'] = texture.Texture2D(mat['map_ks'])
            
        if 'map_normal' in mat:
            mat['normal'] = texture.Texture2D(mat['map_normal'])

        materials[ind] = mat

    return materials
