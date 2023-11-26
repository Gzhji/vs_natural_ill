import configparser
import os
import numpy as np
import open3d as o3d
import pymesh
import copy
import shutil
import fileinput
from pathlib import Path


class AdobeReader:
    def __init__(self):
        super().__init__()
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'

        f = open(base_path + config['adobe']['furn'], 'r')
        lines = [line.strip() for line in f.readlines()]
        f.close()
        self.furns, self.cate, self.front, self.scale = {}, {}, {}, {}
        for line in lines:
            records = line.split()
            furn = records[0]
            cate = records[1]
            self.cate[furn] = cate
            self.front[furn] = '+z'
            self.scale[furn] = 1
            for record in records[2:]:
                if record in ['+x', '-x', '+y', '-y', '+z', '-z']:
                    self.front[furn] = record
                else:
                    self.scale[furn] = float(record)
            if cate not in self.furns:
                self.furns[cate] = []
            self.furns[cate].append(furn)
        self.ori_obj_template = base_path + config['adobe']['obj']
        self.obj_template = base_path + config['adobe']['simp_obj']
        self.mtl_template = base_path + config['adobe']['simp_mtl']
        self.tex_template = base_path + config['adobe']['simp_tex']
        self.norm_obj_template = base_path + config['adobe']['norm_obj']

    def get_furniture_list(self, semantics=None, position=None):
        # free_obj, align_wall, against_wall, on_table, on_wall
        # bed, sofa, shelf, drawer, table, chair, supported, other
        if semantics is 'other':
            furns = self.furns['misc'] + self.furns['againstwall']
            return furns
        elif semantics is not None:
            return self.furns[semantics]
        elif position is not None:
            pos2sem = {'free_obj': ['chair', 'misc'], 'against_wall': ['bed', 'sofa', 'shelf', 'dresser', 'againstwall'], 'align_wall': ['table'], 'on_table': ['ontable'], 'on_wall': ['onwall']}
            furns = []
            for sem in pos2sem[position]:
                furns = furns + self.get_furniture_list(sem)
            return furns
        else:
            furns = []
            for v in self.furns.values():
                furns = furns + v
            return furns


    def load_obj(self, furn, prenormalized=True, meter2scene=1.0, center=[0, 0]):

        if prenormalized:
            mesh = pymesh.load_mesh(self.norm_obj_template.format(*furn.split('/')))
            vertices = copy.deepcopy(mesh.vertices)
            faces = copy.deepcopy(mesh.faces)
            mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                             triangles=o3d.utility.Vector3iVector(faces))
            return mesh

        mesh = pymesh.load_mesh(self.obj_template.format(*furn.split('/')))
        vertices = copy.deepcopy(mesh.vertices)
        faces = copy.deepcopy(mesh.faces)
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                         triangles=o3d.utility.Vector3iVector(faces))
        # scale so that it matches the scene scale
        f = open(self.ori_obj_template.format(*furn.split('/')), 'r')
        unit = f.readline().split()[4]
        f.close()
        if unit == 'inches':
            S = 0.0254
        elif unit == 'meters':
            S = 1
        elif unit == 'centimeters':
            S = 0.01
        elif unit == 'millimeters':
            S = 0.001
        else:
            print(unit)
            assert(False)
        S = S * meter2scene * self.scale[furn]
        mesh.scale(S, center=[0,0,0])

        # rotate so that +z is up, +x is front
        if self.front[furn] == '+x':
            pass
        elif self.front[furn] == '-x':
            R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
            mesh.rotate(R, center=[0,0,0])
        elif self.front[furn] == '+z':
            R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
            mesh.rotate(R, center=[0,0,0])
        elif self.front[furn] == '-z':
            R = mesh.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
            mesh.rotate(R, center=[0,0,0])
        else:
            assert(False)
        R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        mesh.rotate(R, center=[0,0,0])

        # translate so that it sits on plane z=0 and centered at given position
        vertices = np.asarray(mesh.vertices)
        vmin = vertices.min(0)
        vmean = vertices.mean(0)
        T = np.array([center[0]-vmean[0], center[1]-vmean[1], -vmin[2]])
        mesh.translate(T)
        return mesh

    def save_obj(self, mesh, furn, dst_path, fname, symlink=True):
        suffix = Path(furn).name
        # print('suffix',suffix)
        print(f'saving {furn} to {dst_path} {fname}')
        Path(dst_path).mkdir(parents=True, exist_ok=True)


        vertices = np.asarray(mesh.vertices)
        f = open(self.obj_template.format(*furn.split('/')), 'r')
        lines = [line.strip() for line in f.readlines()]
        f.close()
        v_cnt, vn_cnt = 0, 0

        f = open(Path(dst_path) / (fname + '_' + suffix + '.obj'), 'w')
        for line in lines:
            if len(line) == 0:
                pass
            elif line.startswith('mtllib '):
                print('mtllib', fname + '_' + suffix + '.mtl', file=f)
            elif line.startswith('v '):
                print('v', vertices[v_cnt][0], vertices[v_cnt][1], vertices[v_cnt][2], file=f)
                v_cnt = v_cnt + 1
            elif line.startswith('vn '):
                pass
            elif line.startswith('f '):
                print('f ', end='', file=f)
                for i in line[2:].split(' '):
                    print('/'.join(i.split('/')[:2]) + ' ', end='', file=f)
                print(file=f)
            else:
                print(line, file=f)
        f.close()

        mtl_src = Path(self.mtl_template.format(*furn.split('/')))
        print('mtlsrc', mtl_src)
        mtl_dst = Path(dst_path) / (fname + '_' + suffix + '.mtl')


        if mtl_src.exists() and not mtl_dst.exists():
            f = open(mtl_src, 'r')
            lines = [line.strip() for line in f.readlines()]
            f.close()
            f = open(mtl_dst, 'w')
            for line in lines:
                print(line.replace(suffix + '/', fname + '_' + suffix + '/'), file=f)
            f.close()

        tex_src = Path(self.tex_template.format(*furn.split('/')))
        # tex_dst = Path(dst_path) / (fname + '_' + suffix)
        tex_dst = Path(dst_path) / fname
        if tex_src.exists() and not tex_dst.exists():
            if symlink:
                os.symlink(tex_src, tex_dst)
            else:
                shutil.copytree(tex_src, tex_dst)




class BunnyReader:
    def __init__(self, path):
        super().__init__()
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini')
        config = configparser.ConfigParser()
        config.read(config_file)
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
        self.obj_path = path
        self.front = '+x'
        #
        # f = open(base_path + config['adobe']['furn'], 'r')
        # lines = [line.strip() for line in f.readlines()]
        # f.close()
        # self.furns, self.cate, self.front, self.scale = {}, {}, {}, {}
        # for line in lines:
        #     records = line.split()
        #     furn = records[0]
        #     cate = records[1]
        #     self.cate[furn] = cate
        #     self.front[furn] = '+z'
        #     self.scale[furn] = 1
        #     for record in records[2:]:
        #         if record in ['+x', '-x', '+y', '-y', '+z', '-z']:
        #             self.front[furn] = record
        #         else:
        #             self.scale[furn] = float(record)
        #     if cate not in self.furns:
        #         self.furns[cate] = []
        #     self.furns[cate].append(furn)
        # self.ori_obj_template = base_path + config['adobe']['obj']
        # self.obj_template = base_path + config['adobe']['simp_obj']
        # self.mtl_template = base_path + config['adobe']['simp_mtl']
        # self.tex_template = base_path + config['adobe']['simp_tex']
        # self.norm_obj_template = base_path + config['adobe']['norm_obj']




    def load_obj(self, furn, meter2scene=1.0, center=[0, 0]):


        mesh = pymesh.load_mesh(self.obj_path)
        vertices = copy.deepcopy(mesh.vertices)
        faces = copy.deepcopy(mesh.faces)
        mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices),
                                         triangles=o3d.utility.Vector3iVector(faces))
        # scale so that it matches the scene scale

        # rotate so that +z is up, +x is front
        if self.front == '+x':
            pass
        elif self.front == '-x':
            R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
            mesh.rotate(R, center=[0,0,0])
        elif self.front == '+z':
            R = mesh.get_rotation_matrix_from_xyz((0, np.pi/2, 0))
            mesh.rotate(R, center=[0,0,0])
        elif self.front == '-z':
            R = mesh.get_rotation_matrix_from_xyz((0, -np.pi/2, 0))
            mesh.rotate(R, center=[0,0,0])
        else:
            assert(False)
        R = mesh.get_rotation_matrix_from_xyz((np.pi/2, 0, 0))
        mesh.rotate(R, center=[0,0,0])

        # translate so that it sits on plane z=0 and centered at given position
        vertices = np.asarray(mesh.vertices)
        vmin = vertices.min(0)
        vmean = vertices.mean(0)
        T = np.array([center[0]-vmean[0], center[1]-vmean[1], -vmin[2]])
        mesh.translate(T)
        return mesh

    def save_obj(self, mesh, furn, dst_path, fname, symlink=True):
        suffix = Path(furn).name
        print(f'saving {furn} to {dst_path} {fname}')
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        vertices = np.asarray(mesh.vertices)
        f = open(self.obj_template.format(*furn.split('/')), 'r')
        lines = [line.strip() for line in f.readlines()]
        f.close()
        v_cnt, vn_cnt = 0, 0
        f = open(Path(dst_path) / (fname + '_' + suffix + '.obj'), 'w')
        for line in lines:
            if len(line) == 0:
                pass
            elif line.startswith('mtllib '):
                print('mtllib', fname + '_' + suffix + '.mtl', file=f)
            elif line.startswith('v '):
                print('v', vertices[v_cnt][0], vertices[v_cnt][1], vertices[v_cnt][2], file=f)
                v_cnt = v_cnt + 1
            elif line.startswith('vn '):
                pass
            elif line.startswith('f '):
                print('f ', end='', file=f)
                for i in line[2:].split(' '):
                    print('/'.join(i.split('/')[:2]) + ' ', end='', file=f)
                print(file=f)
            else:
                print(line, file=f)
        f.close()

        mtl_src = Path(self.mtl_template.format(*furn.split('/')))
        # mtl_dst = Path(dst_path) / (fname + '_' + suffix + '.mtl')
        mtl_dst = Path(dst_path) / (fname + '.mtl')
        if mtl_src.exists() and not mtl_dst.exists():
            f = open(mtl_src, 'r')
            lines = [line.strip() for line in f.readlines()]
            f.close()
            f = open(mtl_dst, 'w')
            for line in lines:
                #print(line.replace(suffix + '/', fname + '_' + suffix + '/'), file=f)
                print(line.replace(suffix + '/', fname + '/'), file=f)
            f.close()

        tex_src = Path(self.tex_template.format(*furn.split('/')))
        tex_dst = Path(dst_path) / (fname + '_' + suffix)
        if tex_src.exists() and not tex_dst.exists():
            if symlink:
                os.symlink(tex_src, tex_dst)
            else:
                shutil.copytree(tex_src, tex_dst)

if __name__ == '__main__':
    reader = AdobeReader()
    print(reader.get_furniture_list(semantics='other'))

