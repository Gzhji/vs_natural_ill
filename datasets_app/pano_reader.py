import configparser
import os
import torch
import math
import numpy as np
import cv2
import json
from pathlib import Path

def get_reader(dset, height=256, width=None, config_name=None):
    if dset == 'zind':
        reader = ZInDReader(height, width, config_name)
    elif dset == 'laval' or 'hdri' or 'maxim':
        reader = HDRReader(dset, height, width)
    else:
        assert(False)
    return reader


class PanoReader:
    def __init__(self, dset, height, width=None, config_name=None):
        super().__init__()
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.ini' if config_name is None else config_name)
        config = configparser.ConfigParser()
        config.read(config_file)
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/'
        self.template = {}
        for key in config[dset].keys():
            self.template[key] = base_path + config[dset][key]
        self.height = height
        if width is None:
            self.width = height * 2
        else:
            self.width = width


class ZInDReader(PanoReader):

    def __init__(self, height, width=None, config_name=None):
        super().__init__('zind', height, width, config_name)

    def get_rgb_image(self, record_id, key='rgb'):
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        if key in self.template:
            print(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id))
            im = cv2.imread(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id))[:,
                 :, [2, 1, 0]].astype(np.float32) / 255.0
        elif '{' in key:
            im = cv2.imread(key.format(house=house_id, floor=floor_id, pano=pano_id))[:, :,
                 [2, 1, 0]].astype(np.float32) / 255.0
        else:
            im = cv2.imread(key)[:, :, [2, 1, 0]].astype(np.float32) / 255.0
        if (self.height < im.shape[0]):
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2, 0, 1)
        return im

    def get_gray_image(self, record_id, key):
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        print(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id))
        split_id, house_id, floor_id, pano_id = record
        im = cv2.imread(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im)[None]
        return im

    def get_hdr_image(self, record_id, key='hdr'):
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        print(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id))
        im = cv2.imread(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id), cv2.IMREAD_UNCHANGED)[:, :, [2,1,0]].astype(np.float32)
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_hdr_orgi(self, record_id, key='hdr'):
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        im = cv2.imread(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id), cv2.IMREAD_UNCHANGED)[:, :, [2,1,0]].astype(np.float32)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_arch_image(self, record_id, key):
        # 3 x H x W, 0: wall, 1: floor, 2: ceiliing
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        print(self.template[key].format(house=house_id, floor=floor_id, pano=pano_id))
        im = cv2.imread(self.template[key].format(split=split_id, house=house_id, floor=floor_id, pano=pano_id), cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        im[im == 128] = 1
        im[im == 255] = 2
        im = im.astype(np.int64)
        im = torch.from_numpy(im)
        return im

    def get_pos_image(self, record_id):
        # C x H x W
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        im = cv2.imread(self.template['pos'].format(house=house_id, floor=floor_id, pano=pano_id), cv2.IMREAD_UNCHANGED)[:, :, [2,1,0]].astype(np.float32)
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_camera_extrinsics(self, record_id):
        # 6: posx, posy, rot, scale, camh, ceilh
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        json_file = open(self.template['layout'].format(house=house_id, floor=floor_id, pano=pano_id), 'r')
        content = json.load(json_file)
        json_file.close()
        extr = np.asarray([content['position'][0], content['position'][1], content['rotation'], content['scale'], content['camera_height'], content['ceiling_height']], dtype=np.float32)
        extr = torch.from_numpy(extr)
        return extr

    def get_floor_bbox(self, record_id, opendoor=True):
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        if opendoor:
            fname = self.template['floormesh'].format(house=house_id, floor=floor_id, pano=pano_id, modal='geo_floor')
        else:
            fname = self.template['floormesh'].format(house=house_id, floor=floor_id, pano=pano_id, modal='tex_floor')
        f = open(fname, 'r')
        lines = f.readlines()
        f.close()
        vertices = []
        for line in lines:
            if line.startswith('v '):
                x, y, z = line.split()[1:]
                vertices.append([float(x), float(y), float(z)])

        vertices = np.array(vertices)[:,:2]
        bbox = np.stack((vertices.min(0), vertices.max(0)), 0)
        bbox = torch.from_numpy(bbox).float()
        return bbox

    def get_semantic_image(self, record_id, modal='semantic'):
        # H x W
        record = record_id.split(' ')
        house_id, floor_id, pano_id = record
        im = cv2.imread(self.template[modal].format(house=house_id, floor=floor_id, pano=pano_id))[:, :,
             [2, 1, 0]]
        label = np.zeros_like(im[:, :, 0])
        pallete = self._create_color_palette_room()
        for k, v in enumerate(pallete):
            mask = (im[:, :, 0] == v[1][0]) * (im[:, :, 1] == v[1][1]) * (im[:, :, 2] == v[1][2])
            label[mask] = k
        label = cv2.resize(label, (self.width, self.height), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        label = torch.from_numpy(label)
        return label



class HDRReader(PanoReader):

    def __init__(self, dset, height, width=None):
        super().__init__(dset, height, width)

    def get_rgb_image(self, record_id, key='rgb'):
        im = cv2.imread(self.template[key].format(record_id))[:, :, [2,1,0]].astype(np.float32) / 255.0
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im

    def get_hdr_image(self, record_id, key='hdr'):
        im = cv2.imread(self.template[key].format(record_id), cv2.IMREAD_UNCHANGED)[:, :, [2,1,0]].astype(np.float32)
        if self.height < im.shape[0]:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_AREA)
        else:
            im = cv2.resize(im, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        return im
