import os
import sys

import matplotlib.pyplot as plt
import tqdm

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
import copy
import argparse
import json
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import open3d as o3d
import shutil
from scipy.spatial.distance import cdist
from pathlib import Path
from collections import OrderedDict
import mapbox_earcut as earcut
from sympy.geometry import Point, Segment, Polygon
from sympy.geometry.util import convex_hull
from utils_app.transformations import Point2D, Transformation2D, Transformation3D, TransformationSpherical
from utils_app.transform import sph2cart, pix2sph, uvgrid, uv2pos, uv2pix
import json
from utils_app.misc import *
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class ZillowFloor:
    def __init__(self, floor_data):
        self._load_floor(floor_data)

    def _load_floor(self, floor_data):
        self.pano_data = {}
        vertices_list = floor_data['layoutPoints']['points']
        vertices = []
        for vertice in vertices_list:
            # convert from led2net coordinate to zind coordinate
            vertices.append([-vertice['xyz'][0], vertice['xyz'][2]])

        vertices = np.array(vertices)
        walls = self._build_walls(vertices)
        self.pano_data['ceiling_height'] = floor_data['layoutHeight']
        self.pano_data['camera_height'] = floor_data['cameraHeight']
        self.pano_data['vertices'] = vertices
        self.pano_data['walls'] = walls

        corners = []
        for wall_id, wall in enumerate(walls):
            corners.append(wall[0])
        corners = np.stack(corners, axis=0)
        self.pano_data['corners'] = corners


    def _build_walls(self, vertices, eps=1e-2):
        walls = []
        for vert_id in range(len(vertices)):
            vert_p1 = vertices[vert_id]
            vert_p2 = vertices[(vert_id + 1) % len(vertices)]
            walls.append([vert_p1, vert_p2])
        walls = np.array(walls).astype(np.float64) # Nx2x2

        return walls

    def _E2P(self, image, top, bottom, resolution=1024, mode='bilinear'):
        pano_t = torch.from_numpy(image).permute(2, 0, 1)[None]
        small_3d = np.stack((top, bottom), 0)
        small_3d_t = torch.from_numpy(small_3d).permute(2, 0, 1)[None]
        large_3d_t = F.interpolate(small_3d_t, (resolution + 1, resolution + 1), mode='bilinear', align_corners=True)
        large_3d_t = F.avg_pool2d(large_3d_t, 2, 1)
        large_3d = large_3d_t[0].permute(1, 2, 0).numpy()
        large_3d[..., 1] = -large_3d[..., 1]
        large_sph = TransformationSpherical.cartesian_to_sphere(large_3d.reshape(-1, 3)).reshape(resolution, resolution, 3)
        grid = torch.from_numpy(large_sph[:,:,:2])[None].float()
        grid[:, :, :, 0] = grid[:, :, :, 0] / math.pi
        grid[:, :, :, 1] = -grid[:, :, :, 1] / math.pi * 2
        tex_t = F.grid_sample(pano_t, grid, padding_mode='border', mode=mode)
        tex = tex_t[0].permute(1, 2, 0).numpy()

        return tex

    def _P2E(self, tex, image, top, bottom, resolution=1024, mode='bilinear'):
        top_l, top_r, bottom_l, bottom_r = top[0], top[1], bottom[0], bottom[1]
        p2, p1 = top[0, :2], top[1, :2]
        # camera to wall distance
        c2w_distance = np.cross(p2-p1, -p1) / np.linalg.norm(p2 - p1)

        normal = -np.cross(bottom_l-top_l,top_l- top_r )
        normal = normal / np.linalg.norm(normal)
        normal = torch.Tensor(normal)[None,:, None, None]

        uv = uvgrid(resolution * 2, resolution, 1)
        pix = uv2pix(uv)
        sph = pix2sph(pix)
        sph = torch.cat([sph, torch.ones_like(sph[:, 0:1, ...])], dim=1)
        pano_xyz = sph2cart(sph)


        cos_angle = torch.sum(pano_xyz * normal, dim=1, keepdim=True)
        vis_torch_im((cos_angle + 1)/2)
        # vis_torch_im((np.cos(sph[:, 1:2, ...]) + 1)/2)
        pano_xyz = pano_xyz * c2w_distance / (cos_angle * np.cos(sph[:, 1:2, ...]))

        bottom_l = torch.Tensor(bottom_l)[None, :, None, None]
        top_l = torch.Tensor(top_l)[None, :, None, None]
        pano_xyz = (pano_xyz - bottom_l) / (top_l - bottom_l)
        # pano_xyz =
        tex = np2torch_im(tex)
        pano = F.grid_sample(tex, pano_xyz, padding_mode='border', mode=mode)
        pano = pano[0].permute(1, 2, 0).numpy()
        print('pano_coord', pano)



        valid = (pano_xyz[:, 2:3, ...] > bottom[0, 2]) & (pano_xyz[:, 2:3, ...] < top[0, 2])
        vis_torch_im(valid.float())

        max1 = np.maximum(-bottom[0, 1], -bottom[1, 1])
        min1 = np.minimum(-bottom[0, 1], -bottom[1, 1])
        valid1 =  (pano_xyz[:, 1:2, ...] > min1) & (pano_xyz[:, 1:2, ...] < max1)

        max2 = np.maximum(bottom[0, 0], bottom[1, 0])
        min2 = np.minimum(bottom[0, 0], bottom[1, 0])
        valid2 = (pano_xyz[:, 0:1, ...] > min2) & (pano_xyz[:, 0:1, ...] < max2)


        valid = valid & valid1
        plt.imshow(valid.float()[0, 0, ..., None] * image / 255)
        plt.show()
        valid = valid & valid1 & valid2


    def _crop_floor_tex(self, pano_im, corners, transform3d, resolution=1024):
        corners_min, corners_max = corners.min(axis=0), corners.max(axis=0)
        top = np.array([[corners_min[0], corners_min[1]], [corners_max[0], corners_min[1]]])
        bottom = np.array([[corners_min[0], corners_max[1]], [corners_max[0], corners_max[1]]])
        floor_top, ceiling_top = transform3d.to_3d(top)
        floor_bottom, ceiling_bottom = transform3d.to_3d(bottom)
        floor_tex = self._E2P(pano_im, floor_top, floor_bottom, resolution=resolution)
        ceiling_tex = self._E2P(pano_im, ceiling_top, ceiling_bottom, resolution=resolution)
        return floor_tex, ceiling_tex

    def build_tex(self, im_template='panos/{}.jpg', tex_pano=None, pano_im=None, res=1024, wall_ids=None, verbose=False, mode='bilinear'):
        # load and process image
        if im_template is not None:
            pano_im = cv2.imread(im_template, -1)[:,:,[2,1,0]]
            dtype = pano_im.dtype
            pano_im = pano_im.astype(np.float32)
        else:
            dtype = pano_im.dtype
            pano_im = pano_im.astype(np.float32)

        wall_tex, floor_tex, ceiling_tex = [], [], []

        # coordinate transformation
        transform3d = Transformation3D(self.pano_data['ceiling_height'], camera_height=self.pano_data['camera_height'])

        # get the 2D vertices of wall at the top-down view
        if wall_ids is None:
            walls = self.pano_data['walls']
        else:
            wall_ids = np.array(wall_ids)
            walls = self.pano_data['walls'][wall_ids]

        # transform 2D wall vertices to 3D cartesian points.
        floor, ceiling = transform3d.to_3d(walls)

        # build wall texture
        for wall_id in range(len(floor)):
            # equirectangular to perspective
            tex = self._E2P(pano_im, ceiling[wall_id], floor[wall_id], resolution=res, mode=mode)
            if verbose:
                plt.imshow(pano_im)
                plt.show()
                plt.imshow(tex)
                plt.show()
            wall_tex.append(tex)

        # build floor and ceiling texture
        if tex_pano is None:
            # per-room texture
            corners = self.pano_data['corners']
            cur_floor_tex, cur_ceiling_tex = self._crop_floor_tex(pano_im, corners, transform3d, resolution=4096)
            floor_tex.append(cur_floor_tex)
            ceiling_tex.append(cur_ceiling_tex)

        # some post-processing
        wall_tex = np.concatenate(wall_tex, axis=1)
        wall_tex = wall_tex.reshape(-1).reshape(wall_tex.shape).astype(dtype)
        floor_tex = np.concatenate(floor_tex, axis=1)
        floor_tex = floor_tex.reshape(-1).reshape(floor_tex.shape).astype(dtype)
        ceiling_tex = np.concatenate(ceiling_tex, axis=1)
        ceiling_tex = ceiling_tex.reshape(-1).reshape(ceiling_tex.shape).astype(dtype)

        return wall_tex, floor_tex, ceiling_tex


    def get_wall_id_for_division(self, tex_pano, res, thresh=0.1):
        '''
        Dividing the wall into two categories, containing and not containing "window".
        '''
        walls = self.pano_data['walls']
        print('len(walls)' ,len(walls))
        print('tex_pano.shape', tex_pano.shape)
        assert len(walls) == tex_pano.shape[1] // res
        bright_wall_ids = []
        for wall_id in range(len(walls)):
            cur_tex = tex_pano[:, wall_id * res : (wall_id+1) * res, :] > thresh
            if np.sum(cur_tex) > 0:
                bright_wall_ids.append(wall_id)
        dark_wall_ids = [e for e in [i for i in range(len(walls))] if e not in bright_wall_ids]
        return bright_wall_ids, dark_wall_ids

    def _triangulate_poly(self, poly):
        # poly: N x 2
        verts = poly.astype(np.float32)
        rings = np.array([len(verts)], dtype=np.uint32)
        triangle = earcut.triangulate_float32(verts, rings).reshape(-1, 3)
        return triangle

    def _create_trimesh(self, vertices, faces):
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                         triangles=o3d.utility.Vector3iVector(faces))
        return mesh

    def _build_wall_mesh(self, wall_tex, wall_ids = None):
        ## walls
        # wall geometry
        n_walls = 0
        vertices, faces = [], []
        # Make the floor plane to be 0
        transform3d = Transformation3D(ceiling_height=self.pano_data['ceiling_height'], camera_height=0)

        if wall_ids is None:
            walls = self.pano_data['walls']
        else:
            wall_ids = np.array(wall_ids)
            walls = self.pano_data['walls'][wall_ids]
        walls_floor, walls_ceiling = transform3d.to_3d(walls)
        for wall_id in range(len(walls)):
            vertices += [walls_floor[wall_id, 0], walls_ceiling[wall_id, 0], walls_ceiling[wall_id, 1], walls_floor[wall_id, 1]]
            triangle = np.array([[0, 2, 1], [2, 0, 3]])
            faces.append(triangle + n_walls * 4)
            n_walls += 1

        faces = np.concatenate(faces, axis=0).astype(np.int32)
        wall_mesh = self._create_trimesh(vertices, faces)

        # wall texture
        i = 0
        uvs = []
        # walls = self.pano_data['walls']
        for wall_id in range(len(walls)):
            uv = np.array([[i / n_walls, 1], [i / n_walls, 0], [(i+1) / n_walls, 0], [(i+1) / n_walls, 1]], dtype=np.float32)
            uvs.append(uv)
            i += 1
        uvs = np.concatenate(uvs, axis=0)
        if wall_tex is not None:
            wall_mesh.textures = [o3d.geometry.Image(wall_tex)]
        wall_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uvs[faces.reshape(-1), :], dtype=np.float64))
        wall_mesh.compute_vertex_normals()
        return wall_mesh

    def _build_floor_mesh(self, floor_tex, ceiling_tex):
        n_verts = 0
        floor_vertices, floor_faces, ceiling_vertices, ceiling_faces = [], [], [], []

        corners = self.pano_data['corners']
        transform3d = Transformation3D(ceiling_height=self.pano_data['ceiling_height'], camera_height=0)
        corners_floor, corners_ceiling = transform3d.to_3d(corners)
        triangle = self._triangulate_poly(corners)
        floor_vertices += list(corners_floor)
        floor_faces.append(triangle + n_verts)
        ceiling_vertices += list(corners_ceiling)
        ceiling_faces.append(triangle[:, ::-1] + n_verts)
        n_verts += len(corners)


        floor_faces = np.concatenate(floor_faces, axis=0)
        floor_mesh = self._create_trimesh(floor_vertices, floor_faces)
        ceiling_faces = np.concatenate(ceiling_faces, axis=0)
        ceiling_mesh = self._create_trimesh(ceiling_vertices, ceiling_faces)

        uvs = []

        corners = self.pano_data['corners']
        corners_min, corners_max = corners.min(axis=0), corners.max(axis=0)
        uv = (corners - corners_min[None]) / (corners_max[None] - corners_min[None])

        uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)

        if floor_tex is not None:
            floor_mesh.textures = [o3d.geometry.Image(floor_tex)]
        floor_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uvs[floor_faces.reshape(-1), :], dtype=np.float64))
        floor_mesh.compute_vertex_normals()
        if ceiling_tex is not None:
            ceiling_mesh.textures = [o3d.geometry.Image(ceiling_tex)]
        ceiling_mesh.triangle_uvs = o3d.utility.Vector2dVector(np.array(uvs[ceiling_faces.reshape(-1), :], dtype=np.float64))
        ceiling_mesh.compute_vertex_normals()
        return floor_mesh, ceiling_mesh

    def _build_floor_mesh_poly(self, floor_tex, ceiling_tex, mesh_pano=None, scale=3):
        n_verts = 0
        floor_vertices, floor_faces, ceiling_vertices, ceiling_faces = [], [], [], []

        corners = self.pano_data['corners'] * scale
        transform3d = Transformation3D(ceiling_height=self.pano_data['ceiling_height'], camera_height=0)
        corners_floor, corners_ceiling = transform3d.to_3d(corners)
        triangle = self._triangulate_poly(corners)
        floor_vertices += list(corners_floor)
        floor_faces.append(triangle + n_verts)
        ceiling_vertices += list(corners_ceiling)
        ceiling_faces.append(triangle[:, ::-1] + n_verts)
        n_verts += len(corners)

        floor_faces = np.concatenate(floor_faces, axis=0)
        floor_mesh = self._create_trimesh(floor_vertices, floor_faces)
        ceiling_faces = np.concatenate(ceiling_faces, axis=0)
        ceiling_mesh = self._create_trimesh(ceiling_vertices, ceiling_faces)

        uvs = []

        corners = self.pano_data['corners'] * scale
        corners_min, corners_max = corners.min(axis=0), corners.max(axis=0)
        uv = (corners - corners_min[None]) / (corners_max[None] - corners_min[None])

        uvs.append(uv)
        uvs = np.concatenate(uvs, axis=0)

        if floor_tex is not None:
            floor_mesh.textures = [o3d.geometry.Image(floor_tex)]
        floor_mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.array(uvs[floor_faces.reshape(-1), :], dtype=np.float64))
        floor_mesh.compute_vertex_normals()
        if ceiling_tex is not None:
            ceiling_mesh.textures = [o3d.geometry.Image(ceiling_tex)]
        ceiling_mesh.triangle_uvs = o3d.utility.Vector2dVector(
            np.array(uvs[ceiling_faces.reshape(-1), :], dtype=np.float64))
        ceiling_mesh.compute_vertex_normals()
        return floor_mesh, ceiling_mesh

    def build_mesh(self, wall_tex, floor_tex, ceiling_tex, no_hole=False, retreat=False, mesh_pano=None, eps=4.5e-3, wall_ids=None):
        wall_mesh = self._build_wall_mesh(wall_tex, wall_ids)

        if not no_hole:
            floor_mesh, ceiling_mesh = self._build_floor_mesh(floor_tex, ceiling_tex)

        else:
            floor_mesh, ceiling_mesh = self._build_floor_mesh_poly(floor_tex, ceiling_tex, mesh_pano)

        if retreat:
            wall_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(wall_mesh.vertices) - eps * np.asarray(wall_mesh.vertex_normals))
            floor_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(floor_mesh.vertices) - eps * np.asarray(floor_mesh.vertex_normals))
            ceiling_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(ceiling_mesh.vertices) - eps * np.asarray(ceiling_mesh.vertex_normals))
        return wall_mesh, floor_mesh, ceiling_mesh



def save_obj(fname, mesh, exr=None):
    print(f'Saving {fname}')
    basename = Path(fname).name
    dirname = Path(fname).parent
    dirname.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(fname + '.obj', mesh)

    if not mesh.has_textures() and Path(fname + '.mtl').exists():
        Path(fname + '.mtl').unlink()
        f = open(fname + '.obj', 'r')
        lines = f.readlines()
        f.close()
        f = open(fname + '.obj', 'w')
        for line in lines:
            if 'mtl' not in line:
                print(line.strip(), file=f)
        f.close()
    else:
        f = open(fname + '.obj', 'r')
        lines = f.readlines()
        f.close()
        f = open(fname + '.obj', 'w')
        for line in lines:
            print(line.replace('_0', '').strip(), file=f)
        f.close()
        f = open(fname + '.mtl', 'r')
        lines = f.readlines()
        f.close()
        f = open(fname + '.mtl', 'w')
        for line in lines:
            print(line.replace('_0', '').strip(), file=f)
        f.close()
        shutil.move(fname + '_0.png', fname + '.png')
    if exr is not None:
        cv2.imwrite(fname + '.exr', exr[::-1,:,::-1])
        if Path(fname + '.png').exists():
            Path(fname + '.png').unlink()
        if Path(fname + '.mtl').exists():
            Path(fname + '.mtl').unlink()

def write_to_world(fname, pano_data):
    dirname = Path(fname).parent
    dirname.mkdir(parents=True, exist_ok=True)
    xml_file = open(fname + '.xml', 'w')
    scale = pano_data['scale']
    x = -pano_data['position'][0]
    y = -pano_data['position'][1]
    z = -pano_data['camera_height']
    rotation = -pano_data['rotation']
    print(f'        <transform name="toWorld">', file=xml_file)
    print(f'            <scale z="{scale}" />', file=xml_file)
    print(f'            <translate x="{x}" y="{y}" z="{z}" />', file=xml_file)
    print(f'            <rotate z="1" angle="{rotation}" />', file=xml_file)
    print(f'        </transform>', file=xml_file)
    xml_file.close()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-path', type=str, default='../00_Data/zind/scenes/layout_merge')
    parser.add_argument('--out-path', type=str, default='../00_Data/zind/scenes/floormesh')
    parser.add_argument('--hdr', action='store_true')
    parser.add_argument('--split', type=str, default='new+penn+bed+07091127')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    print(opt)

    split_id = opt.split

    #out_folder = Path(opt.out_path) / split_id / f'{split_id}+{house_id}+{floor_id}+{pano_id}'
    #json_path = f'{opt.raw_path}/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}/pred.json'
    out_folder = Path(opt.out_path) / f'{split_id}'

    json_path = f'{opt.raw_path}/{split_id}/pred.json'
    layout_json = json.load(open(json_path))

    zf = ZillowFloor(layout_json)


    wall_mesh, floor_mesh, ceiling_mesh = zf.build_mesh(wall_tex=None, floor_tex=None, ceiling_tex=None, no_hole=False, retreat=False)
    save_obj(str(out_folder / 'geo_wall'), wall_mesh)
    save_obj(str(out_folder / 'geo_ceiling'), ceiling_mesh)
    save_obj(str(out_folder / 'geo_floor'), floor_mesh)
    #
    wall_tex, floor_tex, ceiling_tex = zf.build_tex(f'{opt.raw_path}/{split_id}//color.jpg', tex_pano=None)
    wall_mesh, floor_mesh, ceiling_mesh = zf.build_mesh(wall_tex=wall_tex, floor_tex=floor_tex, ceiling_tex=ceiling_tex, no_hole=False)
    save_obj(str(out_folder / 'tex_wall'), wall_mesh)
    save_obj(str(out_folder / 'tex_ceiling'), ceiling_mesh)
    save_obj(str(out_folder / 'tex_floor'), floor_mesh)
    #
    zf = ZillowFloor(layout_json)
    wall_mesh, floor_mesh, ceiling_mesh = zf.build_mesh(wall_tex=None, floor_tex=None, ceiling_tex=None, no_hole=False, retreat=True)
    save_obj(str(out_folder / 'light_wall'), wall_mesh)
    save_obj(str(out_folder / 'light_ceiling'), ceiling_mesh)
    wall_mesh, floor_mesh, ceiling_mesh = zf.build_mesh(wall_tex=None, floor_tex=None, ceiling_tex=None, no_hole=True, retreat=True)
    save_obj(str(out_folder / 'light_floor'), floor_mesh)

    zf = ZillowFloor(layout_json)
    wall_mesh, floor_mesh, ceiling_mesh = zf.build_mesh(wall_tex=None, floor_tex=None, ceiling_tex=None,
                                                        no_hole=False, retreat=True, eps=-4.5e-3)
    save_obj(str(out_folder / 'mask_ceiling'), ceiling_mesh)

    if opt.hdr:
        wall_tex, floor_tex, ceiling_tex = zf.build_tex(f'../data/zind/scenes/light/{split_id}' + '_hdr.exr')
        cv2.imwrite(str(out_folder / 'tex_wall') + '.exr', wall_tex[::-1,:,::-1])
        cv2.imwrite(str(out_folder / 'tex_ceiling') + '.exr', ceiling_tex[::-1,:,::-1])
        cv2.imwrite(str(out_folder / 'tex_floor') + '.exr', floor_tex[::-1,:,::-1])
