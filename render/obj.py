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

def load_obj(filename, clear_ks=True, mtl_override=None):
    model_components, materials = load_glb.load_glb(filename)
    # Load materials
    all_materials = [
        {
            'name' : '_default_mat',
            'bsdf' : 'pbr',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
    all_materials += material.load_mtl(mtl_override, mtrls=materials)

    # load vertices
    vertices, texcoords, normals  = [], [], []
    vertices = model_components[0]['primitives'][0]['vertex_coords']
    texcoords = model_components[0]['primitives'][0]['tex_coords']
    normals = model_components[0]['primitives'][0]['normals']
    
    # load faces
    activeMatIdx = None
    used_materials = []
    faces, tfaces, nfaces, mfaces = [], [], [], []
    faces = model_components[0]['primitives'][0]['triangle_idx']
    tfaces = model_components[0]['primitives'][0]['triangle_idx']
    nfaces = model_components[0]['primitives'][0]['triangle_idx']
    faces = faces.astype(np.int16)
    tfaces = tfaces.astype(np.int16)
    nfaces = nfaces.astype(np.int16)
    mfaces.extend([0]*len(faces))
    for mat in all_materials:
        used_materials.append(mat)
    
    assert len(tfaces) == len(faces) and len(nfaces) == len (faces)

    # Create an "uber" material by combining all textures into a larger texture
    if len(used_materials) > 1:
        uber_material, texcoords, tfaces = material.merge_materials(all_materials, texcoords, tfaces, mfaces)
    else:
        uber_material = used_materials[0]


    vertices = tensor(vertices, dtype=torch.float32)
    vertices = vertices.contiguous()
        
    texcoords = tensor(texcoords, dtype=torch.float32)
    texcoords = texcoords.contiguous()
    
    normals = tensor(normals, dtype=torch.float32)
    normals = normals.contiguous()
    
    # vertices = torch.tensor(vertices, dtype=torch.float32, device='cuda')
    # texcoords = torch.tensor(texcoords, dtype=torch.float32, device='cuda') if len(texcoords) > 0 else None
    # normals = torch.tensor(normals, dtype=torch.float32, device='cuda') if len(normals) > 0 else None
    
    faces = torch.tensor(faces, dtype=torch.int64, device='cuda')
    tfaces = torch.tensor(tfaces, dtype=torch.int64, device='cuda') if texcoords is not None else None
    nfaces = torch.tensor(nfaces, dtype=torch.int64, device='cuda') if normals is not None else None

    return mesh.Mesh(vertices, faces, normals, nfaces, texcoords, tfaces, material=uber_material)

######################################################################################
# Save mesh object to objfile
######################################################################################

def write_obj(folder, mesh, save_material=True):
    obj_file = os.path.join(folder, 'mesh.obj')
    print("Writing mesh: ", obj_file)
    with open(obj_file, "w") as f:
        f.write("mtllib mesh.mtl\n")
        f.write("g default\n")

        v_pos = mesh.v_pos.detach().cpu().numpy() if mesh.v_pos is not None else None
        v_nrm = mesh.v_nrm.detach().cpu().numpy() if mesh.v_nrm is not None else None
        v_tex = mesh.v_tex.detach().cpu().numpy() if mesh.v_tex is not None else None

        t_pos_idx = mesh.t_pos_idx.detach().cpu().numpy() if mesh.t_pos_idx is not None else None
        t_nrm_idx = mesh.t_nrm_idx.detach().cpu().numpy() if mesh.t_nrm_idx is not None else None
        t_tex_idx = mesh.t_tex_idx.detach().cpu().numpy() if mesh.t_tex_idx is not None else None

        print("    writing %d vertices" % len(v_pos))
        for v in v_pos:
            f.write('v {} {} {} \n'.format(v[0], v[1], v[2]))
       
        if v_tex is not None:
            print("    writing %d texcoords" % len(v_tex))
            assert(len(t_pos_idx) == len(t_tex_idx))
            for v in v_tex:
                f.write('vt {} {} \n'.format(v[0], 1.0 - v[1]))

        if v_nrm is not None:
            print("    writing %d normals" % len(v_nrm))
            assert(len(t_pos_idx) == len(t_nrm_idx))
            for v in v_nrm:
                f.write('vn {} {} {}\n'.format(v[0], v[1], v[2]))

        # faces
        f.write("s 1 \n")
        f.write("g pMesh1\n")
        f.write("usemtl defaultMat\n")

        # Write faces
        print("    writing %d faces" % len(t_pos_idx))
        for i in range(len(t_pos_idx)):
            f.write("f ")
            for j in range(3):
                f.write(' %s/%s/%s' % (str(t_pos_idx[i][j]+1), '' if v_tex is None else str(t_tex_idx[i][j]+1), '' if v_nrm is None else str(t_nrm_idx[i][j]+1)))
            f.write("\n")

    if save_material:
        mtl_file = os.path.join(folder, 'mesh.mtl')
        print("Writing material: ", mtl_file)
        material.save_mtl(mtl_file, mesh.material)

    print("Done exporting mesh")
