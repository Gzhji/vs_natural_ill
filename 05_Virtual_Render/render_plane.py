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

    """
    00 Render all wall plane
    """
    wall_plane(opt, wall_bsdf='radiance_emitter', wall_tex = None, maxd=8, samples = 16)

    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/wall_plane.xml'):
            file_present = True
            break

    scene_wall = mi.load_file(f'{opt.log}/wall_plane.xml')
    image_wall = mi.render(scene_wall, spp=256)
    print('image_wall.shape', image_wall.shape)
    mi.util.write_bitmap(f'{opt.log}/wall_plane.png', image_wall ** (1.0 / 2.2))


    """
    01 Render all floor plane
    """
    floor_plane(opt, floor_bsdf='radiance_emitter', floor_tex = None, maxd=8, samples = 16)

    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/floor_plane.xml'):
            file_present = True
            break

    scene_floor = mi.load_file(f'{opt.log}/floor_plane.xml')
    image_floor = mi.render(scene_floor, spp=256)
    print('image_floor.shape', image_floor.shape)
    mi.util.write_bitmap(f'{opt.log}/floor_plane.png', image_floor ** (1.0 / 2.2))


    """
    02 Render facade wall
    """
    wall_mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_wall_mesh'


    render_facade(opt, wall_mesh_dir, maxd=8, samples = 16)

    file_present = False
    while file_present == False:
        if os.path.isfile(f'{opt.log}/facade_plane.xml'):
            file_present = True
            break

    scene_facade = mi.load_file(f'{opt.log}/facade_plane.xml')
    image_facade = mi.render(scene_facade, spp=256)
    print('image_facade.shape', image_facade.shape)
    mi.util.write_bitmap(f'{opt.log}/image_facade.png', image_facade ** (1.0 / 2.2))

    print("--- %s seconds ---" % (time.time() - start_time))
    print('rendering finished')


