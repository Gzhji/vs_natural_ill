import random
import math
import torch
import torch.utils.data as data

from .pano_reader import get_reader


class PanoDataset(data.Dataset):
    def __init__(self, dsets, record_lists, height, modals, repeat=1, rotate=False):
        super().__init__()
        if type(dsets) is str:
            dsets = [dsets]
        if type(record_lists) is str:
            record_lists = [record_lists]
        assert(len(dsets) == len(record_lists))
        self.records = []
        self.readers = {}
        for i, dset in enumerate(dsets):
            self.readers[dset] = get_reader(dset, height)
            if type(record_lists[i]) == list:
                cur_records = record_lists[i]
            else:
                f = open(record_lists[i], 'r')
                cur_records = [i.strip() for i in f.readlines()]
                f.close()
            self.records = self.records + list(zip([dset] * len(cur_records), cur_records))
        self.height = height
        self.width = height * 2
        self.modals = modals
        self.repeat = repeat
        self.rotate = rotate

    def __len__(self):
        return len(self.records) * self.repeat

    def __getitem__(self, index):
        index = index % len(self.records)
        record = self.records[index]
        dset_id, record_id = record
        entry = [index, dset_id, record_id]
        if self.rotate:
            shifts = random.randint(0, self.width-1)
        else:
            shifts = 0
        if 'pos' in self.modals or 'extr' in self.modals:
            theta = torch.Tensor([-shifts / self.width * math.pi * 2])
            ct, st = torch.cos(theta), torch.sin(theta)
            R = torch.stack((torch.cat((ct, -st), 0), torch.cat((st, ct), 0)), 0)
        for modal in self.modals:
            if modal == 'rgb' or modal == 'lowres' or modal == 'diffuse' or modal == 'specular' or modal == 'sunlight' or modal == 'ambi':
                cur = self.readers[dset_id].get_rgb_image(record_id, modal)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'tripodanno' or modal == 'tripod' or modal == 'light' or modal == 'coarsesunlight':
                cur = self.readers[dset_id].get_gray_image(record_id, modal)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'semantic' or modal == 'semantic_arch' or modal == 'semantic_arch_no_tripod':
                cur = self.readers[dset_id].get_semantic_image(record_id, modal)
                cur = torch.cat((cur[:, shifts:], cur[:, :shifts]), 1)
            elif modal == 'hdr':
                cur = self.readers[dset_id].get_hdr_image(record_id)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
            elif modal == 'arch':
                cur = self.readers[dset_id].get_arch_image(record_id)
                cur = torch.cat((cur[:, shifts:], cur[:, :shifts]), 1)
            elif modal == 'pos':
                cur = self.readers[dset_id].get_pos_image(record_id)
                cur = torch.cat((cur[:, :, shifts:], cur[:, :, :shifts]), 2)
                cur = torch.cat((torch.einsum('ahw,ab->bhw', cur[:2], R), cur[2:]), 0)
            elif modal == 'extr':
                cur = self.readers[dset_id].get_camera_extrinsics(record_id)
                cur[2] = cur[2] + theta
            elif modal == 'bbox':
                cur = self.readers[dset_id].get_floor_bbox(record_id)
            elif modal == 'sundircoarse':
                cur = self.readers[dset_id].get_sundir_vector(record_id, coarse=True)
            elif modal == 'sundir':
                cur = self.readers[dset_id].get_sundir_vector(record_id)
            else:
                assert(False)
            entry.append(cur)
        return entry


if __name__ == '__main__':
    dset = PanoDataset('laval', '../lists/laval.txt', 256, ['rgb', 'hdr'])
    print(dset[10])
