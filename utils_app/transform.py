import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import math
import numpy as np
import kornia
from scipy.spatial.transform import Rotation
try:
    import nvdiffrast.torch as dr
except ImportError:
    pass
from .ops import *

def appendz(pos2d, z):
    """
        Append z axis to 2D position
        pos2d: B x 2[xy] x H x W
        z: scalar or B
        ret: B x 3[xyz] x H x W
    """
    if not torch.is_tensor(z):
        z = torch.Tensor([z]).to(pos2d.device)
    posz = z[:,None,None,None] * torch.ones_like(pos2d[:,:1])
    pos = torch.cat((pos2d, posz), 1)
    return pos


def cam2wld(pos, extr, no_trans=False):
    """
        Camera-to-world transformation
        pos:  B x 3[xyz] x H x W
        extr: B x [tx, ty, rot, scale, camh, ceilh]
        ret:  B x 3[xyz] x H x W
    """
    B, C, H, W = pos.shape
    T = extr[:,[0,1,4]][:,None].to(pos.device) # B x 1 x 3
    angle = torch.deg2rad(extr[:,2])
    cost = torch.cos(angle) # B
    sint = torch.sin(angle) # B
    R = torch.stack((cost, sint, -sint, cost), dim=1).reshape(B, 2, 2).to(pos.device) # B x 2 x 2
    S = extr[:, 3][:,None,None].to(pos.device) # B x 1 x 1
    cam = pos.reshape(B, C, -1).transpose(1, 2) # B x HW x 3
    wld = torch.bmm(cam[:,:,:2], R) * S
    wld = torch.cat((wld, cam[:,:,2:]), dim=2)
    if not no_trans:
        wld = wld + T
    wld = wld.transpose(1, 2).reshape(B, 3, H, W)
    return wld


def wld2cam(pos, extr, no_trans=False):
    """
        World-to-camera transformation
        pos:  B x C x H x W
        extr: B x [tx, ty, rot, scale, camh, ceilh]
        ret:  B x C x H x W
    """
    B, C, H, W = pos.shape
    T = extr[:,[0,1,4]][:,None].to(pos.device) # B x 1 x 2
    angle = torch.deg2rad(extr[:,2])
    cost = torch.cos(angle) # B
    sint = torch.sin(angle) # B
    R = torch.stack((cost, sint, -sint, cost), dim=1).reshape(B, 2, 2).to(pos.device) # B x 2 x 2
    S = extr[:, 3][:,None,None].to(pos.device) # B x 1 x 1
    wld = pos.reshape(B, C, -1).transpose(1, 2)
    if not no_trans:
        wld = wld - T
    cam = torch.bmm((wld[:,:,:2] / S), R.transpose(1, 2))
    cam = torch.cat((cam, wld[:,:,2:]), dim=2)
    cam = cam.transpose(1, 2).reshape(B, 3, H, W)
    return cam

def sun_angle2tri(azimuth, elevation):
    vec = torch.Tensor([ math.cos(azimuth * math.pi / 180), math.sin(azimuth * math.pi / 180),
                        -math.tan(elevation * math.pi / 180)])
    return vec

def cart2sph(cart, eps=1e-6):
    """
        Cartesian-to-spherical transformation
        cart: B x 3[xyz] x H x W
        ret:  B x 3[ThetaPhiRho] x H x W
    """
    x, y, z = cart[:,0], cart[:,1], cart[:,2]
    theta = torch.atan2(x, y)
    rho = torch.linalg.norm(cart, dim=1)
    phi = torch.asin(z / torch.clamp(rho, min=eps))
    sph = torch.stack((theta, phi, rho), dim=1)
    return sph

def sph2cart(sph):
    """
        Spherical-to-cartesian transformation
        sph: B x 3[ThetaPhiRho] x H x W
        ret: B x 3[xyz] x H x W
    """
    theta, phi, rho = sph[:,0], sph[:,1], sph[:,2]
    x = rho * torch.cos(phi) * torch.sin(theta)
    y = rho * torch.cos(phi) * torch.cos(theta)
    z = rho * torch.sin(phi)
    cart = torch.stack((x, y, z), dim=1)
    return cart

def sph2pix(sph):
    """
        Spherical-to-pixel transformation
        sph: B x 3[ThetaPhiRho] x H x W
        pix: B x 2[xy] x H x W, range -1..1
    """
    theta, phi = sph[:,0], sph[:,1]
    x = theta / math.pi
    y = - phi / math.pi * 2
    pix = torch.stack((x, y), dim=1)
    return pix

def pix2sph(pix):
    """
        Pixel-to-spherical transformation
        pix: B x 2[xy] x H x W, range -1..1
        ret: B x 3[ThetaPhi] x H x W
    """
    x, y = pix[:,0], pix[:,1]
    theta = x * math.pi
    phi = - y * math.pi / 2
    sph = torch.stack((theta, phi), dim=1)
    return sph

def cam2pix(pos):
    """
        Camera-to-pixel transformation for spherical camera
    """
    sph = cart2sph(pos)
    pix = sph2pix(sph)
    return pix


def pos2uv(pos, bbox):
    """
        pos:  B x 2[xy] x H x W
        bbox: B x 2[MinMax] x 2[xy]
        ret:  B x 2[uv] x H x W
    """
    uv = (pos[:,:2] - bbox[:,0,:,None,None]) / (bbox[:,1,:,None,None] - bbox[:,0,:,None,None])
    return uv


def uv2pos(uv, bbox):
    """
        uv:  B x 2[uv] x H x W
        bbox: B x 2[MinMax] x 2[xy]
        ret:  B x 2[xy] x H x W
    """
    pos = uv * (bbox[:,1,:,None,None] - bbox[:,0,:,None,None]) + bbox[:,0,:,None,None]
    return pos


def uvgrid(ures, vres=None, batch=1):
    """
        ret: batch x 2[uv] x vres x ures
    """
    if vres is None:
        vres = ures
    theta = torch.Tensor([[1,0,0],[0,1,0]])[None].expand(batch, -1, -1)
    size = (batch, 1, vres, ures)
    grid = F.affine_grid(theta, size, align_corners=False)
    uv = (grid.permute(0, 3, 1, 2) + 1) / 2
    return uv


def uv2pix(uv):
    pix = uv * 2 - 1
    return pix

def pix2uv(pix):
    uv = (pix + 1) / 2
    return uv


def grid_sample(image, grid, mode='bilinear', pad='border', anti_aliasing=False):
    """
         Wrapper for Pytorch grid_sample
         image: B x C x Hi x Wi, 1 x C x Hi x Wi
         grid:  B x 2[xy] x Ho x Wo, 1 x 2 x Ho x Wo
         ret:   B x C x Ho x Wo
    """
    if anti_aliasing:
        grid_up = F.interpolate(grid, (grid.shape[2] * 2, grid.shape[3] * 2), mode='bilinear')
        result_up = grid_sample(image, grid_up, mode=mode, pad=pad, anti_aliasing=False)
        result = F.interpolate(result_up, (grid.shape[2], grid.shape[3]), mode='bilinear')
        return  result
    if image.shape[0] == 1:
        image = image.expand(grid.shape[0], -1, -1, -1)
    else:
        grid = grid.expand(image.shape[0], -1, -1, -1)
    if pad == 'ones':
        result = F.grid_sample(image - 1, grid.permute(0, 2, 3, 1), mode=mode, padding_mode='zeros', align_corners=False).type(image.dtype) + 1
    else:
        result = F.grid_sample(image, grid.permute(0, 2, 3, 1), mode=mode, padding_mode=pad, align_corners=False).type(image.dtype)
    return result


def grid_sample_dr(image, grid, mode='linear-mipmap-linear', pad='clamp', pano=False):
    """
        Wrapper for NVDiffRast texture sample
         image: B x C x Hi x Wi, 1 x C x Hi x Wi
         grid:  B x 2[xy] x Ho x Wo, 1 x 2 x Ho x Wo
        ret: B x C x Ho x Wo
    """
    if image.shape[0] == 1:
        image = image.expand(grid.shape[0], -1, -1, -1)
    else:
        grid = grid.expand(image.shape[0], -1, -1, -1)
    B, C, H, W = image.shape
    uv = (grid + 1) / 2
    uv[:,:,1] = 1 - uv[:,:,1]
    if pano:
        grad = spatial_gradient_pano(uv)
    else:
        grad = kornia.spatial_gradient(uv)
    grad[:,:,1] = -grad[:,:,1]
    uv_da = grad.reshape(grid.shape[0], 4, grid.shape[2], grid.shape[3])
    result = dr.texture(image.permute(0, 2, 3, 1).contiguous(), uv.permute(0, 2, 3, 1).contiguous(), uv_da.permute(0, 2, 3, 1).contiguous(), filter_mode=mode).permute(0, 3, 1, 2)
    return result


def grid_sample_discrete(image, grid, pad='border'):
    """
         Wrapper for Pytorch grid_sample
         image: B x Hi x Wi, 1 x Hi x Wi
         grid:  B x 2[xy] x Ho x Wo, 1 x 2 x Ho x Wo
         ret:   B x Ho x Wo
    """
    image = image[:,None]
    result = grid_sample(image.float(), grid, mode='nearest', pad=pad).type(image.dtype)[:,0]
    return result

def pano2tex_grid(bbox, extr=None, res=512, flip_xy =False):
    """
        Panorama to floor texture grid
    """
    batch = len(bbox)
    uv = uvgrid(res, res, batch).to(bbox.device)
    pos = uv2pos(uv, bbox)
    pos = appendz(pos, 0)
    if extr is not None:
        pos = wld2cam(pos, extr)

    # align coordinate
    if flip_xy:
        pos[:, :2, ...] = -pos[:, :2, ...]
    pix = cam2pix(pos)
    return pix


def tex2pano_grid(pos, bbox, extr):
    """
        Floor texture to panorama grid
        pos: B x 3[xyz] x H x W, local position
    """
    pos = cam2wld(pos, extr)
    pix = uv2pix(pos2uv(pos, bbox))
    return pix


def pano2pers_grid(height, width, fov, pitch, yaw, device):
    tanhalfh = math.tan(fov * math.pi / 180 / 2)
    tanhalfv = tanhalfh * height / width
    top = np.array([[-tanhalfh, 1, tanhalfv], [tanhalfh, 1, tanhalfv]])
    bottom = np.array([[-tanhalfh, 1, -tanhalfv], [tanhalfh, 1, -tanhalfv]])
    small_3d = np.stack((top, bottom), 0)
    small_3d = Rotation.from_euler('xyz', [-pitch, 0, yaw], degrees=True).apply(small_3d.reshape(-1, 3)).reshape(2, 2, 3)
    small_3d = torch.from_numpy(small_3d[None]).permute(0, 3, 1, 2).float().to(device)
    large_3d = F.interpolate(small_3d, (height + 1, width + 1), mode='bilinear', align_corners=True)
    large_3d = F.avg_pool2d(large_3d, 2, 1)
    pix = cam2pix(large_3d)
    return pix

def sun_wall2floor_grid(dirs, pos, extr=None, wpos=None):
    """
        Directional Light warp grid
        T = S + k L, S: floor, T: wall
        dirs (L): N x 3, S->T direction
        pos (S): 1 x C x H x W
        wpos: 2 x W
        ret: sampling grid: N x 2 x H x W
    """
    if extr is not None:
        pos = cam2wld(pos, extr)
    L = dirs
    S = pos
    if wpos is None:
        wpos = pos[0, :2, pos.shape[2]//2, :] # 2 x W
    fx = pos[0, 0]
    fy = pos[0, 1]
    wx = wpos[0]
    wy = wpos[1]
    Dx = wx[None,None,:] - fx[:,:,None]
    Dy = wy[None,None,:] - fy[:,:,None] # H x W x W
    Lx = dirs[:,0]
    Ly = dirs[:,1]
    dot = (Lx[:,None,None,None] * Dx[None] + Ly[:,None,None,None] * Dy[None]) / torch.clamp((Dx**2 + Dy**2)**0.5, min=1e-10)[None] # N x H x W x W
    values, indices = dot.max(dim=3)
    Sxy = pos[:,:2] # 1 x 2 x H x W
    Txy = wpos[:, indices].permute(1,0,2,3) # N x 2 x H x W
    k = torch.linalg.norm(Sxy - Txy, dim=1) # N x H x W
    T = S + k[:,None] * L[:,:,None,None]
    if extr is not None:
        T = wld2cam(T, extr.expand(T.shape[0], -1))
    pix = cam2pix(T)
    return pix


def sun_floor2wall_grid(dirs, pos, extr=None):
    """
        Directional Light warp grid
        T = S + k L, S: wall, T: floor
        dirs (L): N x 3, S->T direction
        pos (S): 1 x C x H x W
        ret: sampling grid: N x 2 x H x W
    """
    if extr is not None:
        pos = cam2wld(pos, extr)
    Tz = pos[:,2,-1,:].mean()
    L = dirs
    S = pos
    k = (Tz - S[:,2:]) / L[:,2:,None,None] # N x 1 x H x W
    T = S + k * L[:,:,None,None]
    if extr is not None:
        T = wld2cam(T, extr.expand(T.shape[0], -1))
    pix = cam2pix(T)
    return pix


def sun_wall2wall_grid(dirs, pos, extr=None, wpos=None):
    """
        Directional Light warp grid
        T = S + k L
        dirs (L): N x 3, S->T direction
        pos (S): 1 x C x H x W
        wpos: 2 x W
        ret: sampling grid:   N x 2 x H x W
             S validity: N x 1 x 1 x W
    """
    if extr is not None:
        pos = cam2wld(pos, extr)
    L = dirs
    S = pos
    if wpos is None:
        wpos = pos[0, :2, pos.shape[2]//2, :] # 2 x W
    wx = wpos[0]
    wy = wpos[1]
    Dx = wx[None,:] - wx[:,None] # W x W (S x T)
    Dy = wy[None,:] - wy[:,None] # W x W
    Lx = dirs[:,0]
    Ly = dirs[:,1]
    dot = (Lx[:,None,None] * Dx[None] + Ly[:,None,None] * Dy[None]) / torch.clamp((Dx**2 + Dy**2)**0.5, min=1e-10)[None] # N x W x W
    values, indices = dot.max(dim=2)      # N x W
    valid = (values > 0.99).float()[:, None, None, :] # N x 1 x 1 x W
    Sxy = wpos[None]                  # 1 x 2 x W
    Txy = wpos[:, indices].permute(1,0,2) # N x 2 x W
    k = torch.linalg.norm(Sxy - Txy, dim=1) # N x W
    T = S + k[:,None,None] * L[:,:,None,None]
    if extr is not None:
        T = wld2cam(T, extr.expand(T.shape[0], -1))
    pix = cam2pix(T)
    return pix, valid

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
    from datasets_app.pano_reader import get_reader
    from misc import *
    reader = get_reader('zillow')
    record = 'sample 000 floor_01 pano_002'
    extr = reader.get_camera_extrinsics(record)[None]
    rgb = reader.get_rgb_image(record)[None]
    cam = reader.get_pos_image(record)[None]
    fmask = reader.get_fmask_image(record)[None]
    bbox = reader.get_floor_bbox(record)[None]


    uv = uvgrid(512,512,1)
    pos = uv2pos(uv, bbox)
    pos = appendz(pos, 0)
    local = wld2cam(pos, extr)
    pix = cam2pix(local)

    save_im('tex1.png', grid_sample(rgb,  pix))
    save_im('mask1.png', grid_sample(fmask,  pix))
