import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import pyshtools

from . import util
from . import renderutils as ru

######################################################################################
# Utility functions
######################################################################################

class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap):
        return util.avg_pool_nhwc(cubemap, (2,2))

    @staticmethod
    def backward(ctx, dout):
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"), 
                                    torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                                    indexing='ij')
            v = util.safe_normalize(util.cube_to_dir(s, gx, gy))
            out[s, ...] = dr.texture(dout[None, ...] * 0.25, v[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        return out

######################################################################################
# Split-sum environment map light source with automatic mipmap generation
######################################################################################

class EnvironmentLight(torch.nn.Module):
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(self, base):
        super(EnvironmentLight, self).__init__()
        self.mtx = None      
        self.base = torch.nn.Parameter(base.clone().detach(), requires_grad=True)
        self.register_parameter('env_base', self.base)

    def xfm(self, mtx):
        self.mtx = mtx

    def clone(self):
        return EnvironmentLight(self.base.clone().detach())

    def clamp_(self, min=None, max=None):
        self.base.clamp_(min, max)

    def get_mip(self, roughness):
        return torch.where(roughness < self.MAX_ROUGHNESS
                        , (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS) / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) * (len(self.specular) - 2)
                        , (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS) / (1.0 - self.MAX_ROUGHNESS) + len(self.specular) - 2)
        
    def build_mips(self, cutoff=0.99):
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = ru.diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS) + self.MIN_ROUGHNESS
            self.specular[idx] = ru.specular_cubemap(self.specular[idx], roughness, cutoff) 
        self.specular[-1] = ru.specular_cubemap(self.specular[-1], 1.0, cutoff)

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))

    def shade(self, gb_pos, gb_normal, kd, ks, view_pos, specular=True):
        wo = util.safe_normalize(view_pos - gb_pos)

        if specular:
            roughness = ks[..., 1:2] # y component
            metallic  = ks[..., 2:3] # z component
            spec_col  = (1.0 - metallic)*0.04 + kd * metallic
            diff_col  = kd * (1.0 - metallic)
        else:
            diff_col = kd

        reflvec = util.safe_normalize(util.reflect(wo, gb_normal))
        nrmvec = gb_normal
        if self.mtx is not None: # Rotate lookup
            mtx = torch.as_tensor(self.mtx, dtype=torch.float32, device='cuda')
            reflvec = ru.xfm_vectors(reflvec.view(reflvec.shape[0], reflvec.shape[1] * reflvec.shape[2], reflvec.shape[3]), mtx).view(*reflvec.shape)
            nrmvec  = ru.xfm_vectors(nrmvec.view(nrmvec.shape[0], nrmvec.shape[1] * nrmvec.shape[2], nrmvec.shape[3]), mtx).view(*nrmvec.shape)

        # Diffuse lookup
        diffuse = dr.texture(self.diffuse[None, ...], nrmvec.contiguous(), filter_mode='linear', boundary_mode='cube')
        shaded_col = diffuse * diff_col

        if specular:
            # Lookup FG term from lookup texture
            NdotV = torch.clamp(util.dot(wo, gb_normal), min=1e-4)
            fg_uv = torch.cat((NdotV, roughness), dim=-1)
            if not hasattr(self, '_FG_LUT'):
                self._FG_LUT = torch.as_tensor(np.fromfile('/home2/hitesh.goel/3D-modelling/diffusion-materials/thirdparty/differentiableRenderer/data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
            fg_lookup = dr.texture(self._FG_LUT, fg_uv, filter_mode='linear', boundary_mode='clamp')

            # Roughness adjusted specular env lookup
            miplevel = self.get_mip(roughness)
            spec = dr.texture(self.specular[0][None, ...], reflvec.contiguous(), mip=list(m[None, ...] for m in self.specular[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

            # Compute aggregate lighting
            reflectance = spec_col * fg_lookup[...,0:1] + fg_lookup[...,1:2]
            shaded_col += spec * reflectance

        return shaded_col * (1.0 - ks[..., 0:1]) # Modulate by hemisphere visibility
    
    def phong_shading(self, gb_pos, gb_normal, kd, ks, view_pos, shininess=1):
        wo = util.safe_normalize(view_pos - gb_pos)
        light_dir = util.safe_normalize(view_pos - gb_pos)  # Corrected the direction of light

        light_color = torch.tensor([1.0,1.0,1.0], device=kd.device, dtype=kd.dtype)  # White light color

        # Ambient term (constant color contribution)
        ambient_color = kd * 0.1  # Assuming global ambient light intensity of 0.1
        ambient = ambient_color

        # Diffuse term (matte appearance)
        diffuse_intensity = torch.clamp(torch.sum(gb_normal * light_dir, dim=-1), min=0)
        diffuse_color = kd * diffuse_intensity.unsqueeze(-1)
        diffuse = diffuse_color * light_color

        # Specular term (shiny highlights)
        reflvec = util.safe_normalize(util.reflect(light_dir, gb_normal))  # Corrected the reflection vector
        specular_intensity = torch.clamp(torch.sum(wo * reflvec, dim=-1), min=0) ** shininess
        specular_color = ks * specular_intensity.unsqueeze(-1)
        specular = specular_color * light_color

        # Final shaded color
        shaded_col = diffuse + specular

        return shaded_col
    
    
    def schlick_approximation(self, cos_theta, F0):
        return F0 + (1 - F0) * (1 - cos_theta) ** 5

    def ggx_distribution(self, NdotH, roughness):
        alpha = roughness ** 2
        denom = (NdotH ** 2) * (alpha ** 2 - 1) + 1
        return alpha ** 2 / (3.14 * denom ** 2)

    def cook_torrance_shading(self, gb_pos, gb_normal, kd, ks, view_pos, ior=1.0):
        roughness = ks[..., 1:2]
        wo = util.safe_normalize(view_pos - gb_pos)
        light_dir = util.safe_normalize(view_pos - gb_pos)
        light_color = torch.tensor([1.0, 1.0, 1.0], device=kd.device, dtype=kd.dtype)  # White light color

        # Ambient term (constant color contribution)
        ambient_color = kd * 0.1  # Assuming global ambient light intensity of 0.1
        ambient = ambient_color

        # Diffuse term (Lambertian reflection)
        diffuse_intensity = torch.clamp(torch.sum(gb_normal * light_dir, dim=-1), min=0)
        diffuse_color = kd * diffuse_intensity.unsqueeze(-1)  # Broadcasting corrected
        diffuse = diffuse_color * light_color

        # Cook-Torrance specular term (GGX microfacet reflection)
        NdotH = torch.clamp(torch.sum(gb_normal * util.safe_normalize(light_dir + wo), dim=-1), min=0)
        VdotH = torch.clamp(torch.sum(wo * util.safe_normalize(light_dir + wo), dim=-1), min=0)

        F0 = ((ior - 1) / (ior + 1)) ** 2
        F = self.schlick_approximation(VdotH, F0)
        D = self.ggx_distribution(NdotH, roughness)
        G = torch.min(1.0, 2 * NdotH * VdotH / VdotH)  # Schlick-Smith approximation for G

        specular_intensity = (F * D * G) / (4 * VdotH * NdotH)
        specular_color = ks * specular_intensity.unsqueeze(-1)  # Broadcasting corrected
        specular = specular_color * light_color

        # Final shaded color
        shaded_col = ambient + diffuse + specular

        return shaded_col
######################################################################################
# Load and store
######################################################################################

# Load from latlong .HDR file
def _load_env_hdr(fn, scale=1.0):
    latlong_img = torch.tensor(util.load_image(fn), dtype=torch.float32, device='cuda')*scale
    cubemap = util.latlong_to_cubemap(latlong_img, [512, 512])

    l = EnvironmentLight(cubemap)
    l.build_mips()

    return l

def load_env(fn, scale=1.0):
    if os.path.splitext(fn)[1].lower() == ".hdr":
        return _load_env_hdr(fn, scale)
    else:
        assert False, "Unknown envlight extension %s" % os.path.splitext(fn)[1]

def save_env_map(fn, light):
    assert isinstance(light, EnvironmentLight), "Can only save EnvironmentLight currently"
    if isinstance(light, EnvironmentLight):
        color = util.cubemap_to_latlong(light.base, [512, 1024])
    util.save_image_raw(fn, color.detach().cpu().numpy())

######################################################################################
# Create trainable env map with random initialization
######################################################################################

def create_trainable_env_rnd(base_res, scale=0.5, bias=0.25):
    base = torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device='cuda') * scale + bias
    return EnvironmentLight(base)
      
