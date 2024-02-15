import numpy as np
import mitsuba as mi
import drjit as dr
mi.set_variant("cuda_ad_rgb")
import cv2
import matplotlib.pyplot as plt
import argparse
from renderer_helper import *
import time

def hdr2ldr(hdr_img, ratio = 1):
    hdr_img = hdr_img * ratio
    # Simply clamp values to a 0-1 range for tone-mapping
    ldr = np.clip(hdr_img, 0, 1)
    # Color space conversion
    ldr = ldr ** (1 / 2.2)
    # 0-255 remapping for bit-depth conversion
    ldr = 255.0 * ldr
    return ldr

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



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--house', type=str, default='penn')
    parser.add_argument('--floor', type=str, default='bed')
    parser.add_argument('--pano', type=str, default='07091127')
    parser.add_argument('--log', type=str, default='out_obj')
    parser.add_argument('--cache', type=str, default='obj_cache')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--mask_samples', type=int, default=256)
    parser.add_argument('--tex_res', type=int, default=128)
    parser.add_argument('--ldsampler', type=int, default=512)

    parser.add_argument('--camera_x', type=float, default=1)
    parser.add_argument('--camera_y', type=float, default=1)
    parser.add_argument('--camera_z', type=float, default=1)

    parser.add_argument('--lamp_x', type=float, default=-0.5)
    parser.add_argument('--lamp_y', type=float, default=-0.7)
    parser.add_argument('--lamp_z', type=float, default=-1.5)
    parser.add_argument('--spd_file_path', type=str, default='render_night/Calculated Data 4 LED with yellow LED.spd')

    opt = parser.parse_args()
    opt.cache = f'{opt.cache}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.cache, exist_ok=True)
    opt.log = f'{opt.log}/{opt.house}+{opt.floor}+{opt.pano}'
    os.makedirs(opt.log, exist_ok=True)
    return opt


if __name__ == '__main__':
    start_time = time.time()
    opt = parse_args()
    print(opt)

    bright_rt_wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_rt_wall_mesh'
    dark_rt_wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_dark_rt_wall_mesh'
    bright_wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_wall_mesh'

    envmap_hdr_path = f'../00_Data/zind/scenes/env_map/{opt.house}+{opt.floor}+{opt.pano}.hdr'
    #envmap_hdr_path = f'../data/zind/scenes/env_map/{opt.split}/2F_W(K)_10141349_1655.hdr'
    floor_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_floor.png'
    ceil_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_ceiling.png'
    wall_tex_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_wall.png'
    wall_mesh_dir = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/tex_wall'
    wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/ambi_wall_mesh_rt'

    """
    00 Render night 
    """
    render_night(opt, envmap_hdr_path, wall_mesh_dir,
                      bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, furn=True, maxd=8,
                      floor_bsdf='Wooden_Parquet_Floor', wall_bsdf='White_Painted_Wall', ceil_bsdf='White_Painted_Ceil', furn_bsdf='full',
                      floor_tex = floor_tex_path, wall_tex = wall_tex_path, ceil_tex = ceil_tex_path,
                      samples = 16, wall_extend = True, factor=20, retreat=False)

    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/scene_night.xml'):
            file_present = True
            break

    scene_night = mi.load_file(f'{opt.log}/scene_night.xml')
    image_night = mi.render(scene_night, spp=256)
    mi.util.write_bitmap(f'{opt.log}/scene_night.png', image_night ** (1.0 / 2.2))

    print("--- %s seconds ---" % (time.time() - start_time))
    print('rendering finished')


