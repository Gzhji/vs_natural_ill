import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import argparse
import sh
import time
from pathlib import Path
import numpy as np
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import math
import lxml.etree as ET
from xml.dom import minidom
from datasets_app.pano_reader import get_reader
from utils_app.misc import *
from utils_app.transform import *
from utils_app.mtlparser import MTLParser
from utils_app.ops import *
from renderer_helper import *

import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")



def unit_angle_z(v):
    temp = dr.asin(0.5 * dr.norm(mi.Vector3f(v.x, v.y, v.z - dr.mulsign(mi.Float(1.0), v.z)))) * 2
    return dr.select(v.z >= 0, temp, dr.pi - temp)

class SphericalCamera(mi.Sensor):
    """Defines a spherical sensor that is used for a figure"""

    def __init__(self, props=mi.Properties()):
        super().__init__(props)
        self.use_sphere_uv = props.get('use_sphere_uv', False)

    def sample_ray(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        wavelengths, wav_weight = self.sample_wavelengths(dr.zeros(mi.SurfaceInteraction3f),
                                                          wavelength_sample, active)
        o = self.world_transform().translation()

        if self.use_sphere_uv:
            d = self.world_transform() @ mi.warp.square_to_uniform_sphere(position_sample)
        else:
            sin_phi, cos_phi = dr.sincos(2 * dr.pi * position_sample.x)
            sin_theta, cos_theta = dr.sincos(dr.pi * position_sample.y)
            d = self.world_transform() @ mi.Vector3f(sin_phi * sin_theta, cos_theta, -cos_phi * sin_theta)
        return mi.Ray3f(o, d, time, wavelengths), wav_weight

    def sample_ray_differential(self, time, wavelength_sample, position_sample, aperture_sample, active=True):
        ray, weight = self.sample_ray(time, wavelength_sample, position_sample, aperture_sample, active)
        return mi.RayDifferential3f(ray), weight

    def sample_direction(self, it, sample, active=True):
        # Transform the reference point into the local coordinate system
        trafo = self.world_transform()
        ref_p = trafo.inverse() @ it.p
        d = mi.Vector3f(ref_p)
        dist = dr.norm(d)
        inv_dist = 1.0 / dist
        d *= inv_dist
        resolution = self.film().crop_size()

        ds = dr.zeros(mi.DirectionSample3f)

        if self.use_sphere_uv:
            theta = unit_angle_z(d)
            phi = dr.atan2(d.y, d.x)
            phi[phi < 0.0] += 2 * dr.pi
            ds.uv = mi.Point2f(phi * dr.inv_two_pi, theta * dr.inv_pi)
            ds.uv.x -= dr.floor(ds.uv.x)
            ds.uv *= resolution
            sin_theta = dr.safe_sqrt(1 - d.z * d.z)
        else:
            ds.uv = mi.Point2f(dr.atan2(d.x, -d.z) * dr.inv_two_pi, dr.safe_acos(d.y) * dr.inv_pi)
            ds.uv.x -= dr.floor(ds.uv.x)
            ds.uv *= resolution
            sin_theta = dr.safe_sqrt(1 - d.y * d.y)

        ds.p = trafo.translation()
        ds.d = (ds.p - it.p) * inv_dist
        ds.dist = dist
        ds.pdf = dr.select(active, 1.0, 0.0)

        weight = (1 / (2 * dr.pi * dr.pi * dr.maximum(sin_theta, dr.epsilon(mi.Float)))) * dr.sqr(inv_dist)
        return ds, mi.Spectrum(weight)

mi.register_sensor("spherical", lambda props: SphericalCamera(props))


def Flip_Img(img):
    h, w = img.shape[:2]
    print('h, w; ', h,w)
    l_img = img[0:h, 0:w//2]
    r_img = img[0:h, w//2:w]
    concatenated_img = np.concatenate((r_img, l_img), axis=1)
    return concatenated_img

def Resize(img, target_h):
    dim = (target_h*2, target_h)
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return new_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='coda')#coda+609+03121400
    parser.add_argument('--floor', type=str, default='609')
    parser.add_argument('--pano', type=str, default='03121400')
    parser.add_argument('--log', type=str, default='out_obj')
    parser.add_argument('--cache', type=str, default='obj_cache')
    parser.add_argument('--layout-path', type=str, default='../00_Data/zind/scenes/layout_merge')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--mask_samples', type=int, default=256)
    parser.add_argument('--tex_res', type=int, default=128)

    opt = parser.parse_args()
    opt.cache = f'{opt.cache}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.cache, exist_ok=True)
    opt.log = f'{opt.log}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.log, exist_ok=True)


    with open(f'{opt.log}/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    return opt


if __name__ == '__main__':

    start_time = time.time()
    opt = parse_args()
    print(opt)

    """
    00: load images
    """
    reader = get_reader('zind', opt.height)
    record = f'{opt.house} {opt.floor} {opt.pano}'
    print('img height for this rendering is:', opt.height)

    rgb_i = reader.get_rgb_image(record, 'rgb')[None]
    win_i = reader.get_rgb_image(record, 'window_mask')[None]
    envmap_hdr = reader.get_hdr_orgi(record, 'env_map_hdr')[None]

    save_im(f'{opt.log}/0_rgb_i.png', rgb_i)
    save_im(f'{opt.log}/0_win_i.png', win_i)
    save_hdr(f'{opt.cache}/0_env_map.hdr', envmap_hdr)

    #camera setting #extr: B x [tx, ty, rot, scale, camh, ceilh]
    extr = np.asarray([0, 0, 0, 1, 1.6, 7], dtype=np.float32)
    extr = torch.from_numpy(extr)[None]

    #load json file from floor segmentation
    json_path = f'{opt.layout_path}/{opt.house}+{opt.floor}+{opt.pano}/pred.json'
    layout_json = json.load(open(json_path))


    zf = ZillowFloor(layout_json)
    # for reproduction
    to_world_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/toworld.xml'
    if os.path.exists(to_world_path):
        tree = ET.parse(to_world_path)
        root = tree.getroot()
        for child in root:
            if child.tag == 'translate':
                ratio = -float(child.attrib['z'])/ extr[0, 4].item()

        for key in zf.pano_data.keys():
            zf.pano_data[key] = zf.pano_data[key] * ratio


    """
    01: Build tex
    """
    # build floor wall tex
    ambi_wall_tex, ambi_floor_tex, _ = zf.build_tex(None, tex_pano=None, pano_im=rgb_i[0].permute(1, 2, 0).numpy(), res=opt.tex_res)
    # save ambi texture for ambi
    ambi_wall_tex = np.flip(ambi_wall_tex, axis=0)
    ambi_wall_tex = torch.Tensor(ambi_wall_tex.copy()).permute(2, 0, 1)[None]
    ambi_wall_tex_dir = f'{opt.cache}/ambi_wall_tex.exr'

    ambi_floor_tex = np.flip(ambi_floor_tex, axis=0)
    ambi_floor_tex = torch.Tensor(ambi_floor_tex.copy()).permute(2, 0, 1)[None]
    ambi_floor_tex_dir = f'{opt.cache}/ambi_floor_tex.exr'

    save_hdr(ambi_wall_tex_dir, ambi_wall_tex)
    save_hdr(ambi_floor_tex_dir, ambi_floor_tex)
    save_im(f'{opt.log}/1_ambi_wall_tex.png', ambi_wall_tex)
    save_im(f'{opt.log}/1_ambi_floor_tex.png', ambi_floor_tex)

    # build mesh
    ambi_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=None, retreat=True, eps=-4.5e-3)
    ambi_wall_mesh_dir = f'{opt.cache}/ambi_wall_mesh'
    save_obj(ambi_wall_mesh_dir, ambi_wall_mesh, )

    # build retreat mesh
    ambi_wall_mesh_rt, _, _ = zf.build_mesh(None, None, None, wall_ids=None, retreat=True, eps=-4.5e-5)
    ambi_wall_mesh_rt_dir = f'{opt.cache}/ambi_wall_mesh_rt'
    save_obj(ambi_wall_mesh_rt_dir, ambi_wall_mesh_rt, )


    """
    02: Build Wall-window Mask
    """
    win_mask_tex, _, _ = zf.build_tex(None, tex_pano=None, pano_im=win_i[0].permute(1, 2, 0).numpy(), res=opt.tex_res)
    win_mask_tex = np.flip(win_mask_tex, axis=0)
    win_mask_tex = torch.Tensor(win_mask_tex.copy()).permute(2, 0, 1)[None]

    win_mask_tex_dir = f'{opt.cache}/win_mask_tex.exr'
    save_hdr(win_mask_tex_dir, win_mask_tex)
    win_mask_tex_img = save_im(f'{opt.log}/1_win_mask_tex.png', win_mask_tex)

    win_mask_tex_img = win_mask_tex_img[:, :, 0]
    win_flip = 255 - win_mask_tex_img
    cv2.imwrite(f'{opt.log}/1_win_mask_tex_img.png', win_flip)


    """
    03: find bright and dark walls
    """
    bright_wall_ids, dark_wall_ids = zf.get_wall_id_for_division(torch2np_im(win_mask_tex), 128)
    print('bright_wall_ids is:', bright_wall_ids)

    # build texture (diff to ambi ratio)
    dark_wall_tex, dark_floor_tex, _ = zf.build_tex(None,
        tex_pano=None, pano_im=torch2np_im(rgb_i), res=opt.tex_res, wall_ids=dark_wall_ids)
    bright_wall_tex, _, _ = zf.build_tex(None,
        tex_pano=None, pano_im=torch2np_im(win_i), res=opt.tex_res, wall_ids=bright_wall_ids)
    bright_wall_tex = torch.Tensor(bright_wall_tex).permute(2, 0, 1)[None]
    bright_wall_tex = bright_wall_tex.mean(1, keepdim=True)
    save_im(f'{opt.cache}/3_dark_wall_tex.png', torch.Tensor(dark_wall_tex.copy()).permute(2, 0, 1)[None] / 255)
    save_im(f'{opt.cache}/3_bright_wall_tex.png', bright_wall_tex)

    bright_wall_tex_mask = torch2np_im(bright_wall_tex)
    bright_wall_tex_mask = 255 * (1 - bright_wall_tex_mask)
    bright_wall_tex_mask = bright_wall_tex_mask.astype(float) * 255
    bright_wall_tex_mask = cv2.flip(bright_wall_tex_mask, 0)
    cv2.imwrite(f'{opt.cache}/3_bright_wall_tex_mask.png', bright_wall_tex_mask)



    """
    04: build obj bright and dark walls
    """
    bright_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=bright_wall_ids)
    dark_wall_mesh, _, _ = zf.build_mesh(None , None, None, wall_ids=dark_wall_ids)
    bright_wall_mesh_dir = f'{opt.cache}/3_bright_wall_mesh'
    dark_wall_mesh_dir = f'{opt.cache}/3_dark_wall_mesh'
    save_obj(bright_wall_mesh_dir, bright_wall_mesh, )
    save_obj(dark_wall_mesh_dir, dark_wall_mesh, )

    '''
    05: build obj bright and dark walls retreated
    '''
    bright_rt_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=bright_wall_ids, retreat=True, eps=-4.5e-3)
    dark_rt_wall_mesh, _, _ = zf.build_mesh(None, None, None, wall_ids=dark_wall_ids, retreat=True, eps=-4.5e-3)
    bright_rt_wall_mesh_dir = f'{opt.cache}/3_bright_rt_wall_mesh'
    dark_rt_wall_mesh_dir = f'{opt.cache}/3_dark_rt_wall_mesh'
    save_obj(bright_rt_wall_mesh_dir, bright_rt_wall_mesh, )
    save_obj(dark_rt_wall_mesh_dir, dark_rt_wall_mesh, )

    """
    06: render light on indoor plane using env map
    """
    envmap_hdr_path = f'../00_Data/zind/scenes/env_map/{opt.house}+{opt.floor}+{opt.pano}' + '.hdr'
    floor_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'
    ceil_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.png'
    wall_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_wall.png'
    wall_tex_path = f'{opt.log}/1_win_mask_tex.png'
    wall_mesh_dir = f'{opt.cache}/ambi_wall_mesh_rt'

    #position for hemisphere
    pos = [1, -2, 0]
    Sun2All_xml = render_texture(opt, envmap_hdr_path, wall_mesh_dir, bright_rt_wall_mesh_dir,
                            floor_bsdf = 'white_floor', wall_bsdf='white_wall',
                            ceil_bsdf = 'white_ceiling', furn_bsdf='full',
                            floor_tex= floor_tex_path, wall_tex=wall_tex_path, ceil_tex=ceil_tex_path,
                            maxd=8, furn= False, samples=opt.mask_samples,
                            wall_extend = True, hem_pos = np.array(pos))

    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/scene_hem.xml'):
            file_present = True
            break

    Shading_scene = mi.load_file(f'{opt.log}/scene_hem.xml')
    Shading_scene_image = mi.render(Shading_scene, spp=256)

    mi.util.write_bitmap(f'{opt.log}/shading_layer.png', (Shading_scene_image[::, ::, 0:3])**(1.0 / 2.2))
    mi.util.write_bitmap(f'{opt.log}/shading_layer.exr', Shading_scene_image)
    mi.util.write_bitmap(f'{opt.log}/normal.png', Shading_scene_image[::, ::, 5:8])


    """
    07: instrinsic decomposition
    """
    rgb_name = 'complete_rgb.jpg'
    shading_name = 'shading_layer.png'

    # Wait for 10 seconds to ensure shading image has been saved
    time.sleep(10)
    rgb_img = cv2.imread(f'{opt.log}/' + rgb_name, -1)
    shading_img = cv2.imread(f'{opt.log}/' + shading_name, -1)

    albedo_img = np.array(rgb_img, dtype=np.uint8)
    shading_img = np.array(shading_img, dtype=np.uint8)

    if shading_img.shape[0] != rgb_img.shape[0]:
        shading_img = Resize(shading_img, rgb_img.shape[0])

    albedo_img = rgb_img/shading_img
    albedo_img = albedo_img * 255
    cv2.imwrite(f'{opt.log}/' + 'albedo_img.jpg', albedo_img)

    #bilaberal filter to smooth the result
    albedo_img = np.array(albedo_img, dtype=np.float32)
    img_bi = cv2.bilateralFilter(albedo_img, d=50, sigmaColor=15, sigmaSpace=75)
    cv2.imwrite(f'{opt.log}/' + 'color.jpg', img_bi)
    print('albedo layer has been saved as color.jpg')

    #compute executive time for rendering
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The complete rendering took {elapsed_time} seconds.")
