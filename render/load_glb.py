import os
import pygltflib
from pygltflib import GLTF2
import numpy as np
import trimesh

from PIL import Image
import io

import util
import imageio
import torch

def to_float32_img(img):
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.float32:
        return img
    
    return img

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def apply_transformations(curr_style_dic, ref_pose):
    for node, mesh in curr_style_dic.items():
        if node not in ref_pose:
            print("ERROR: node {} not in ref_pose!".format(node), flush = True)
        elif 'translation' not in ref_pose[node]:
            print("ERROR: translation not in ref_pose[{}]!".format(node), flush = True)
        elif 'rotation' not in ref_pose[node]:
            print("ERROR: rotation not in ref_pose[{}]!".format(node), flush = True)

        translation = trimesh.transformations.translation_matrix(np.asarray(ref_pose[node]['translation'])) # 4, 4
        q = np.asarray(ref_pose[node]['rotation'])
        q = np.roll(q, shift = 1) # x, y, z, w -> w, x, y, z
        rotation = trimesh.transformations.quaternion_matrix(q)        # 4, 4
        scale = np.diag(np.pad(np.asarray(ref_pose[node]['scale']), pad_width = ((0, 1)), constant_values = 1))
        transform = np.matmul(np.matmul(translation, rotation), scale)
        
        for primitive in mesh['primitives']:
            vertices = None
            if primitive['vertex_coords'] is not None:
                vertices = primitive['vertex_coords']
                vertices = np.transpose(np.pad(array = vertices, pad_width = ((0, 0), (0, 1)), mode = 'constant', constant_values = 1)) # 4, 429
                vertices = np.matmul(transform, vertices) # 4, 429
                vertices = np.transpose(vertices)[:, :-1] # 429, 3
                
                # Store transformed vertices
                primitive['vertex_coords'] = vertices

            normals = None
            if primitive['normals'] is not None:
                normals = primitive['normals']
                normals = np.transpose(np.pad(array = normals, pad_width = ((0, 0), (0, 1)), mode = 'constant', constant_values = 1)) # 4, 429
                normals = np.matmul(rotation, normals) # 4, 429
                normals = np.transpose(normals)[:, :-1] # 429, 3
                
                primitive['normals'] = normals

def get_data_type(pygltflib_code):
    if pygltflib_code == pygltflib.UNSIGNED_BYTE:
        return "uint8"
    elif pygltflib_code == pygltflib.UNSIGNED_SHORT:
        return "uint16"
    elif pygltflib_code == pygltflib.UNSIGNED_INT:
        return "uint32"
    elif pygltflib_code == pygltflib.FLOAT:
        return "float32"
    else:
        raise Exception('Unknown data type code {}'.format(pygltflib_code))

def extract_materials(gltf, output_path = None):
    materials = []
    for mat in gltf.materials:
        mat_name = mat.name
        pbrMetallicRoughness = mat.pbrMetallicRoughness
        
        base_color_texture_idx = pbrMetallicRoughness.baseColorTexture.index
        metallic_roughness_texture_idx = pbrMetallicRoughness.metallicRoughnessTexture.index
        normal_texture_idx = mat.normalTexture.index
        
        curr_mat = {}
        materials.append(curr_mat)
        curr_mat['name'] = mat_name
        
        for idx, type in zip([base_color_texture_idx, metallic_roughness_texture_idx, normal_texture_idx], ['base_color', 'roughness', 'normal']):
            texture = extract_texture(gltf, idx)
            
            if type == 'roughness':
                curr_mat['roughness'] = to_float32_img(np.asarray(texture)[..., 1])
                curr_mat['metallic'] = to_float32_img(np.asarray(texture)[..., 2])
            else:
                curr_mat[type] = to_float32_img(np.asarray(texture))
            
            if output_path is not None:
                if texture.format == 'JPEG':
                    ext = '.jpg'
                else:
                    ext = '.png'
                    
                if type == 'roughness':
                    # "Its green channel contains roughness values and its blue channel contains metalness values.""
                    roughness = Image.fromarray((255 * np.asarray(texture)[..., 1]).astype(np.uint8))
                    metalness = Image.fromarray((255 * np.asarray(texture)[..., 2]).astype(np.uint8))
                    roughness.save(os.path.join(output_path, "{}_{}{}".format(mat_name, "roughness", ext)))
                    metalness.save(os.path.join(output_path, "{}_{}{}".format(mat_name, "metallic", ext)))
                else:
                    texture.save(os.path.join(output_path, "{}_{}{}".format(mat_name, type, ext)))
    return materials

def extract_node(gltf, node):
    binary_blob = gltf.binary_blob()
    node_attrs = gltf.nodes[node]

    if node_attrs.mesh is None:
        return None

    node_pos = {}
    if node_attrs.rotation is None:
        node_pos['rotation'] = [1.0, 0.0, 0.0, 0.0] # x, y, z, w
    else:
        node_pos['rotation'] = node_attrs.rotation

    if node_attrs.scale is None:
        node_pos['scale'] = [1.0, 1.0, 1.0] # sx, sy, sz
    else:
        node_pos['scale'] = node_attrs.scale

    if node_attrs.translation is None:
        node_pos['translation'] = [0.0, 0.0, 0.0]
    else:
        node_pos['translation'] = node_attrs.translation

    mesh = gltf.meshes[node_attrs.mesh]

    curr_node_dic = {}
    curr_node_dic['name'] = node_attrs.name
    curr_node_dic['rotation'] = node_attrs.rotation
    curr_node_dic['translation'] = node_attrs.translation
    curr_node_dic['primitives'] = []

    for primitive in mesh.primitives:
        curr_primitive_dic = {}

        if primitive.attributes.TEXCOORD_0 is None:
            print("WARNING: tex_coords are None -> Skipping Primitive!".format(primitive.attributes.TEXCOORD_0), flush = True)
            # print("\tModel Name: {}".format(model_name), flush = True)
            # print("\tStyle Idx: {}".format(style_idx), flush = True)
            continue

        curr_node_dic['primitives'].append(curr_primitive_dic)

        key_map = {
            primitive.indices:               'triangle_idx',
            primitive.attributes.POSITION:   'vertex_coords',
            primitive.attributes.TEXCOORD_0: 'tex_coords',
            primitive.attributes.NORMAL:     'normals'
        }

        curr_primitive_dic['material_idx'] = primitive.material                    
            
        for attr, data_key in key_map.items():
            if attr is None:
                curr_primitive_dic[data_key] = None
                print("WARNING: attr {} is None -> Skipping!".format(data_key), flush = True)
                continue

            accessor = gltf.accessors[attr]
            bufferView = gltf.bufferViews[accessor.bufferView]

            if accessor.type == 'SCALAR': elems_per_entity = 1
            elif accessor.type == 'VEC2': elems_per_entity = 2
            elif accessor.type == 'VEC3': elems_per_entity = 3

            attr_data = np.frombuffer(
                binary_blob[
                    bufferView.byteOffset
                    + accessor.byteOffset : bufferView.byteOffset
                    + bufferView.byteLength
                ],
                dtype = get_data_type(accessor.componentType),
                count = accessor.count * elems_per_entity,
            ).reshape((-1, elems_per_entity))

            if attr == primitive.indices:
                attr_data = attr_data.reshape((-1, 3))

            curr_primitive_dic[data_key] = attr_data

    return node_pos, curr_node_dic

def extract_texture(gltf, texture_idx):
    binary_blob = gltf.binary_blob()
    texture_img = gltf.images[gltf.textures[texture_idx].source] # images[texture_idx]
    bufferView = gltf.bufferViews[texture_img.bufferView]
    raw_data = binary_blob[bufferView.byteOffset : bufferView.byteOffset + bufferView.byteLength]
    return Image.open(io.BytesIO(raw_data))

def load_glb(model_path):
    model_components = {}
    ref_pose = {}
    glb_mesh = GLTF2().load_binary(model_path)
    all_nodes = [y for sublist in [x.nodes for x in glb_mesh.scenes] for y in sublist]

    for node in all_nodes:
        node_data = extract_node(glb_mesh, node)

        if node_data is not None:
            node_pos, curr_node_dic = node_data
            model_components[node] = curr_node_dic

            ref_pose[node] = node_pos
            
    apply_transformations(model_components, ref_pose)  
    # out_path = 'textures/'  
    materials = extract_materials(glb_mesh)
    
    return model_components, materials

# if __name__ == "__main__":
#     model_path = '../data/chair_custom.glb'
#     model_components, materials = load_glb(model_path)
#     ind = -1
#     for k,v in model_components.items():
#         ind += 1
#         # print(k,v['primitives'])
#         a = v['primitives'][0]['triangle_idx']
#         a = a.astype(np.int16)
#         a = tensor(a, dtype=torch.int32) # triangle index
#         a = a.contiguous()
        
#         b = v['primitives'][0]['vertex_coords']
#         b = tensor(b, dtype=torch.float32) # vertex coordinates
#         b = b.contiguous()
#         glctx = dr.RasterizeCudaContext()
        
#         tex_coords = v['primitives'][0]['tex_coords']
#         # tex_coords = np.pad(tex_coords, ((0, 0), (0, 1)), mode='constant', constant_values=0)
#         tex_coords = tensor(tex_coords, dtype=torch.float32)
#         tex_coords = tex_coords.contiguous()
        
#         normals = v['primitives'][0]['normals']
#         normals = tensor(normals, dtype=torch.float32)
#         normals = normals.contiguous()

#         base_color_tex = materials[ind]['base_color']
#         base_color_tex = tensor(base_color_tex, dtype=torch.float32)
#         base_color_tex = base_color_tex.contiguous()
        
#         roughness_tex = materials[ind]['roughness']
#         roughness_tex = tensor(roughness_tex, dtype=torch.float32)
#         roughness_tex = roughness_tex.contiguous()
        
#         metallic_tex = materials[ind]['metallic']
#         metallic_tex = tensor(metallic_tex, dtype=torch.float32)
#         metallic_tex = metallic_tex.contiguous()
        
#         normal_tex = materials[ind]['normal']
#         normal_tex = tensor(normal_tex, dtype=torch.float32)
#         normal_tex = normal_tex.contiguous()
#         break