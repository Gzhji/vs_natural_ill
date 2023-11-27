import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import argparse
import time
from pathlib import Path
import numpy as np
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='penn')
    parser.add_argument('--floor', type=str, default='bed')
    parser.add_argument('--pano', type=str, default='07091127')
    parser.add_argument('--log', type=str, default='out_obj')
    parser.add_argument('--cache', type=str, default='obj_cache')
    parser.add_argument('--layout-path', type=str, default='../00_Data/zind/scenes/layout_merge')
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--mask_samples', type=int, default=256)
    parser.add_argument('--tex_res', type=int, default=128)
    parser.add_argument('--hdr2ldr_ratio', type=int, default=5)
    parser.add_argument('--obj_msk_ratio', type=int, default=100) #pixel for grayscale

    opt = parser.parse_args()
    opt.cache = f'{opt.cache}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.cache, exist_ok=True)
    opt.log = f'{opt.log}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.log, exist_ok=True)

    with open(f'{opt.log}/opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)
    return opt

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

def hdr2ldr(hdr_img, ratio = 2):
    hdr_img = hdr_img * ratio
    # Simply clamp values to a 0-1 range for tone-mapping
    ldr = np.clip(hdr_img, 0, 1)
    # Color space conversion
    ldr = ldr ** (1 / 2.2)
    # 0-255 remapping for bit-depth conversion
    ldr = 255.0 * ldr
    return ldr


def Flip_Img(img):
    h, w = img.shape[:2]
    l_img = img[0:h, 0:w//2]
    r_img = img[0:h, w//2:w]
    concatenated_img = np.concatenate((r_img, l_img), axis=1)
    return concatenated_img

def Resize(img, target_h):
    dim = (target_h*2, target_h)
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return new_img

def get_facade_mask(normal_img, color = 151.0):
    facade_mask = normal_img.copy()
    facade_mask[facade_mask != color] = 1
    facade_mask[facade_mask == color] = 0
    facade_mask = np.repeat(facade_mask[:, :, np.newaxis], 3, axis=2)
    return facade_mask

def reverse_mask(img):
    img = ~img.astype(bool)
    img = img.astype(int)
    return img

if __name__ == '__main__':
    start_time = time.time()
    opt = parse_args()
    print('rendering parameters are:', opt)

    """
    step 0 - using the data inputs and floor textures 
    """
    #the path for the data inputs
    envmap_hdr_path = f'../00_Data/zind/scenes/env_map/{opt.house}+{opt.floor}+{opt.pano}' + '.hdr'
    floor_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'
    ceil_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.png'
    wall_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_wall.png'

    #the path for the textures and object mesh
    wall_mask_path = f'../03_Reflectance_Tex/{opt.log}/1_win_mask_tex_img.png'
    wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/ambi_wall_mesh_rt'
    bright_rt_wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_rt_wall_mesh'
    bright_wall_tex = f'../03_Reflectance_Tex/{opt.cache}/3_bright_wall_tex.png'
    dark_rt_wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_dark_rt_wall_mesh'

    #position for hemisphere
    pos = np.array([0, 0, 0])

    #generate xml file for rendering
    Sun2All_xml = render_texture(opt, envmap_hdr_path, bright_wall_tex,
                            wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                            floor_bsdf = 'Wooden_Parquet_Floor', wall_bsdf='White_Painted_Wall',
                            ceil_bsdf = 'White_Painted_Ceil', furn_bsdf='full',
                            floor_tex= floor_tex_path, wall_tex=wall_tex_path, ceil_tex=ceil_tex_path,
                            maxd=8, furn = True, samples=opt.mask_samples,
                            wall_extend = True, hem_pos = pos)

    #check whether the xml file existed
    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/scene_hem.xml'):
            file_present = True
            break

    #render with mitsuba
    render_file = mi.load_file(f'{opt.log}/scene_hem.xml')
    render_scene = mi.render(render_file, spp=256)
    mi.util.write_bitmap(f'{opt.log}/rendered.png', render_scene**(1.0 / 2.2))
    mi.util.write_bitmap(f'{opt.log}/rendered.hdr', render_scene)

    # Wait for 10 seconds to ensure rendering to be saved
    time.sleep(10)
    print('rendering is done')


    """
    step 1 - transfer window region from original image to the rendered image 
    """
    #read window mask from the local folder
    win_mask_path = f'../00_Data/zind/scenes/win_mask/{opt.house}+{opt.floor}+{opt.pano}' + '+win.png'
    win_img = cv2.imread(win_mask_path, cv2.IMREAD_GRAYSCALE)
    win_mask = get_facade_mask(win_img, color=255)
    win_mask = Resize(win_mask, opt.height)
    rev_facade2 = reverse_mask(win_mask)

    #import the rendered img
    rend_ldr = cv2.imread(f'{opt.log}/rendered.png')
    rend_ldr = Resize(rend_ldr, opt.height)

    #import the original input img
    input_image = cv2.imread(f'../03_Reflectance_Tex/{opt.log}/0_rgb_i.png')
    input_image = Resize(input_image, opt.height)

    #base rendering image without window
    rend_ldr_base = rend_ldr * win_mask
    cv2.imwrite(f'{opt.log}/rend_ldr_base.png', rend_ldr_base)

    #window region from the original image
    win_region = input_image * rev_facade2
    cv2.imwrite(f'{opt.log}/win_region.png', win_region)

    #combine rendering with original window
    concate_scene = win_region + rend_ldr_base
    cv2.imwrite(f'{opt.log}/concate_scene.png', concate_scene)
    print('window transfer is done')


    """
    step 2 - editing floor texture
    """
    #generate xml file for rendering
    Sun2obj_xml = render_obj_mask(opt, envmap_hdr_path, bright_wall_tex,
                            wall_mesh_dir, bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir,
                            floor_bsdf = 'white', wall_bsdf='white',
                            ceil_bsdf = 'white', furn_bsdf='black',
                            floor_tex= floor_tex_path, wall_tex=wall_tex_path, ceil_tex=ceil_tex_path,
                            maxd=8, furn = True, samples=opt.mask_samples,
                            wall_extend = True, hem_pos = pos)

    #check whether the xml file existed
    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/scene_obj.xml'):
            file_present = True
            break

    #render with mitsuba
    render_file = mi.load_file(f'{opt.log}/scene_obj.xml')
    render_scene = mi.render(render_file, spp=256)
    mi.util.write_bitmap(f'{opt.log}/obj_mask.png', render_scene**(1.0 / 2.2))
    mi.util.write_bitmap(f'{opt.log}/obj_mask.exr', render_scene)

    # Wait for 10 seconds to ensure rendering to be saved
    time.sleep(10)
    print('rendering obj mask is done')


    """
    step 3 - post processing for floor texture
    """
    
    # import images
    render_img = concate_scene
    obj_mask = cv2.imread(f'{opt.log}/obj_mask.png', 0)
    flr_mask = cv2.imread(f'../03_Reflectance_Tex/{opt.log}/floor_mask.png', 0)
    flr_mask = Resize(flr_mask, opt.height)
    hdr_img = cv2.imread(f'{opt.log}/rendered.hdr', -1)

    #tone mapping hdr to ldr
    hdr_ratio = opt.hdr2ldr_ratio
    rendered_ldr = hdr2ldr(hdr_img, ratio = hdr_ratio)
    cv2.imwrite(f'{opt.log}/hdr2ldr_%02d.png'% hdr_ratio, rendered_ldr)

    # open floor texture is floor boundary substract object coverage
    obj_mask[obj_mask > opt.hdr2ldr_ratio] = 255
    cv2.imwrite(f'{opt.log}/obj_mask.png', obj_mask)
    target_flr = obj_mask / 2 + flr_mask / 2

    # get binarized floor and non-floor texture masks
    target_flr[target_flr != 255] = 0
    cv2.imwrite(f'{opt.log}/target_flr.png', target_flr)
    non_flr = 255 - target_flr
    cv2.imwrite(f'{opt.log}/non_flr.png', non_flr)

    # the floor texture is removed from original rendered img
    non_flr = non_flr / 255
    non_flr_tex = render_img * np.repeat(non_flr[..., np.newaxis], 3, axis=2)
    cv2.imwrite(f'{opt.log}/non_flr_tex.png', non_flr_tex)

    # floor texture from tone mapped hdr
    ldr_flr_tex = rendered_ldr
    target_flr = target_flr / 255
    flr_only_tex = ldr_flr_tex * np.repeat(target_flr[..., np.newaxis], 3, axis=2)
    cv2.imwrite(f'{opt.log}/seg_flr_Tex.png', flr_only_tex)

    # save concatenated final image
    all = non_flr_tex + flr_only_tex
    cv2.imwrite(f'{opt.log}/all_render.png', all)

    # check how object's boundary is outlined
    obj_mask = 0.5 * np.repeat(obj_mask[..., np.newaxis], 3, axis=2)
    test_img = (obj_mask / 255) * render_img
    cv2.imwrite(f'{opt.log}/test_img.png', test_img)

    #compute executive time for rendering
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The complete rendering took {elapsed_time} seconds.")