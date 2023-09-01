import numpy as np
import torch

from render import util
from render import mesh
from render import render
from render import light

# from .dataset import Dataset

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class MyMesh():

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]

        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        self.ref_mesh = mesh.compute_tangents(ref_mesh)
        self.mv, self.mvp, self.campos, self.iter_res, self.iter_spp = self._rotate_scene()

    def _rotate_scene(self):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (3/4 ) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp
    
    def get_output(self):
        rast, out = render.render_mesh(self.glctx, self.ref_mesh, self.mvp, self.campos, self.envlight, self.iter_res, spp=self.iter_spp, 
                                num_layers=self.FLAGS.layers, msaa=True, background=None)
        img = out['shaded']

        return {
            'mv' : self.mv,
            'mvp' : self.mvp,
            'campos' : self.campos,
            'resolution' : self.iter_res,
            'spp' : self.iter_spp,
            'img' : img,
            'rast': rast
        }
