# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import torch

from . import texture
from . import mesh
from . import material
from . import load_glb
import numpy as np

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

######################################################################################
# Utility functions
######################################################################################

def _find_mat(materials, name):
    for mat in materials:
        if mat['name'] == name:
            return mat
    return materials[0] # Materials 0 is the default

######################################################################################
# Create mesh object from objfile
######################################################################################

def load_obj(filename, ind, clear_ks=True, mtl_override=None):
    model_components, materials = load_glb.load_glb(filename)
    
    all_materials = []
    all_materials = material.load_mtl(mtl_override, mtrls=materials, ind=ind)

    # load vertices
    vertices, texcoords, normals  = [], [], []
    vertices = model_components[ind]['primitives'][0]['vertex_coords']
    texcoords = model_components[ind]['primitives'][0]['tex_coords']
    normals = model_components[ind]['primitives'][0]['normals']

    # load faces
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    faces = model_components[ind]['primitives'][0]['triangle_idx']
    tfaces = model_components[ind]['primitives'][0]['triangle_idx']
    nfaces = model_components[ind]['primitives'][0]['triangle_idx']
    faces = faces.astype(np.int16)
    tfaces = tfaces.astype(np.int16)
    nfaces = nfaces.astype(np.int16)
    mfaces.extend([0]*len(faces))
    for mat in all_materials:
        used_materials.append(mat)

    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)
    
    uber_material = used_materials[0]

    vertices = tensor(vertices, dtype=torch.float32)
    vertices = vertices.contiguous()
        
    texcoords = tensor(texcoords, dtype=torch.float32)
    texcoords = texcoords.contiguous()
    
    normals = tensor(normals, dtype=torch.float32)
    normals = normals.contiguous()
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    return mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material)