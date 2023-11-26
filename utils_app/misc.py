import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import shutil
import math
import time


ARCH_WALL = 0
ARCH_FLOOR = 1
ARCH_CEILING = 2

SEM_NONE = 0
SEM_FLOOR = 1
SEM_CEILING = 2
SEM_WALL = 3
SEM_DOOR = 4
SEM_WINDOW = 5
SEM_LIGHT = 6
SEM_PROP = 7
SEM_STRUCT = 8


def th2np(th):
    return th.detach().cpu().numpy()


def sk2cv(sk):
    if len(sk.shape) == 2:
        return sk
    if sk.shape[0] == 1:
        return sk.transpose((1, 2, 0))
    elif sk.shape[0] == 2:
        cv = sk.transpose((1, 2, 0))[:, :, [1, 0]]
        cv = np.concatenate((np.zeros_like(cv[:, :, :1]), cv), 2)
        return cv
    elif sk.shape[0] == 3:
        return sk.transpose((1, 2, 0))[:, :, [2, 1, 0]]
    elif sk.shape[0] == 4:
        return sk.transpose((1, 2, 0))[:, :, [2, 1, 0, 3]]
    else:
        assert(False)


def th2cv(th):
    return sk2cv(th2np(th))

def torch2np_im(im):
    return im.cpu()[0].permute(1,2,0).numpy()

def np2torch_im(im, is_cuda=False):
    return torch.Tensor(im).permute(2, 0, 1)[None]

def rgb2gray(rgb):
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    return  gray


def rgb2luma(rgb):
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return  luma


def rgb2luma_torch(rgb):
    luma = 0.2126 * rgb[:, 0:1] + 0.7152 * rgb[:, 1:2] + 0.0722 * rgb[:, 1:2]
    return  luma

def read_im(fname):
    im = cv2.imread(fname, -1)
    if len(im.shape) == 2:
        im = im[:,:,None]
    else:
        im = im[:,:,[2,1,0]]
    if im.dtype == np.uint8:
        im = im.astype(np.float32) / 255.0
    im = torch.from_numpy(im).permute(2,0,1)[None]
    return im

def srgb2lin(im):
    return torch.clamp(im, min=0) ** 2.2

def lin2srgb(im):
    return torch.clamp(im, min=0) ** (1/2.2)


def to_device(th_list, device):
    return [i.to(device) for i in th_list]

def remkdir(directory):
    shutil.rmtree(directory, ignore_errors=True)
    Path(directory).mkdir(parents=True)


def print_log(log, fname=None, screen=True, mode='a'):
    if type(log) is list:
        with open(fname, mode) as f:
            print(*log, file=f)
        if screen:
            print(*log)
    else:
        with open(fname, mode) as f:
            print(log, file=f)
        if screen:
            print(log)


class Logger():
    def __init__(self, log_path, task, stage, epoch, data_len, lossnames):
        self.log_path = Path(log_path)
        self.task = task
        self.stage = stage
        self.epoch = epoch
        self.data_len = data_len
        self.lossnames = lossnames
        self.start_time = time.time()
        self.total_loss = []

    def log_step(self, iteration, batch_size, losses):
        self.total_loss.append([batch_size])
        logst = f'{self.task} {self.stage} [{self.epoch}]({iteration}/{self.data_len}) {time.time()-self.start_time:.2f}s'
        self.start_time = time.time()
        for lossname in self.lossnames:
            if lossname in losses:
                logst += f' {lossname}:{losses[lossname].item():.6f}'
                self.total_loss[-1].append(losses[lossname].item())
        print_log(logst, self.log_path / 'log.txt', screen=True)

    def log_epoch(self):
        total_loss = np.array(self.total_loss).sum(0)
        total_loss = total_loss[1:] / total_loss[0]
        print_log(list(total_loss), self.log_path / f'{self.stage}_loss.txt', screen=True)


def read_list(fname, pattern=None):
    f = open(fname, 'r')
    lines = [line.strip() for line in f.readlines()]
    f.close()
    if pattern is not None:
        selected = []
        for line in lines:
            if line.find(pattern) != -1:
                selected.append(line)
        return selected
    else:
        return lines


def save_im(fname, im, overwrite=True):
    fpath = Path(fname)
    if not overwrite and fpath.exists():
        return
    fpath.parents[0].mkdir(parents=True, exist_ok=True)
    im = th2cv(torch.clamp(im[0].float(), 0, 1) * 255)
    cv2.imwrite(str(fname), im)
    return im


def save_hdr(fname, im, overwrite=True):
    fpath = Path(fname)
    if not overwrite and Path(fname).exists():
        return
    fpath.parents[0].mkdir(parents=True, exist_ok=True)
    im = th2cv(im[0])
    cv2.imwrite(str(fname), im)
    return im


def save_vis_old(vis_path, data, types, pref, suffs=None, overwrite=True, subdir=False):
    """
        types: ['im', 'hdr', 'srgb']
    """
    if suffs is None:
        suffs = types
    ret = []
    for i, item in enumerate(data):
        if subdir:
            save_path = Path(vis_path) / suffs[i]
        else:
            save_path = Path(vis_path)
        if types[i] == 'im':
            cur = save_im(save_path / f'{pref}_{suffs[i]}.png', item, overwrite)
        elif types[i] == 'hdr':
            cur = save_hdr(save_path / f'{pref}_{suffs[i]}.exr', item, overwrite)
        elif types[i] == 'srgb':
            cur = save_im(save_path / f'{pref}_{suffs[i]}.png', item[:1] ** (1/2.2), overwrite)
        else:
            assert(False)
        ret.append(cur)
    return ret

def save_vis(vis_path, data, pref, overwrite=True, subdir=False):
    """
        data: (im, vtype, suff)
    """
    for i, item in enumerate(data):
        im, vtype, suff = item
        if subdir:
            save_path = Path(vis_path) / suff
        else:
            save_path = Path(vis_path)
        if vtype == 'im':
            cur = save_im(save_path / f'{pref}_{suff}.png', im, overwrite)
        elif vtype == 'hdr':
            cur = save_hdr(save_path / f'{pref}_{suff}.exr', im, overwrite)
        elif vtype == 'srgb':
            cur = save_im(save_path / f'{pref}_{suff}.png', im[:1] ** (1/2.2), overwrite)
        else:
            assert(False)

def vis_torch_im(im):
    if im.shape[1] == 1:
        plt.imshow(im[0, 0].cpu())
        plt.show()
    else:
        plt.imshow(im[0].permute(1, 2, 0).cpu())
        plt.show()

def label2color(label, palette='room'):
    # B x H x W label
    B, H, W = label.shape
    if palette == 'room':
        palette = create_color_palette_room()
    elif palette == 'nyu':
        palette = create_color_palette_nyu()
    elif palette == 'ade':
        palette = create_color_palette_ade()
    elif palette == 'nyu2room':
        label = convert_label(label, palette)
        palette = create_color_palette_room()
    elif palette == 'ade2nyu':
        label = convert_label(label, palette)
        palette = create_color_palette_nyu()
    elif palette == 'ade2room':
        label = convert_label(label, palette)
        palette = create_color_palette_room()
    else:
        assert(False)
    color = torch.zeros(B, 3, H, W).to(label.device)
    for i in range(len(palette)):
        mask = (label == i)
        color[:, 0][mask] = palette[i][1][0]
        color[:, 1][mask] = palette[i][1][1]
        color[:, 2][mask] = palette[i][1][2]
    return color / 255.0


def convert_label(label, convert):
    if convert == 'nyu2room':
        convert_map = map_nyu2room()
        target_palette = create_color_palette_room()
    elif convert == 'ade2nyu':
        convert_map = map_ade2nyu()
        target_palette = create_color_palette_nyu()
    elif convert == 'ade2room':
        convert_map = map_ade2room()
        target_palette = create_color_palette_room()
    else:
        assert(False)
    name2id = {}
    for i in range(len(target_palette)):
        name2id[target_palette[i][0]] = i
    converted = torch.zeros_like(label)
    for i in range(len(convert_map)):
        mask = (label == i)
        target = convert_map[i][1]
        converted[mask] = name2id[target]
    return converted


def convert_prob(prob, convert):
    if convert == 'nyu2room':
        convert_map = map_nyu2room()
        target_palette = create_color_palette_room()
    elif convert == 'ade2nyu':
        convert_map = map_ade2nyu()
        target_palette = create_color_palette_nyu()
    elif convert == 'ade2room':
        convert_map = map_ade2room()
        target_palette = create_color_palette_room()
    else:
        assert(False)
    
    name2id = {}
    for i in range(len(target_palette)):
        name2id[target_palette[i][0]] = i

    B, _, H, W = prob.shape
    C = len(target_palette)
    converted = torch.zeros(B, C, H, W, device=prob.device)
    for i in range(len(convert_map)):
        tid = name2id[convert_map[i][1]]
        converted[:, tid] = converted[:, tid] + prob[:, i]
    return converted


def create_color_palette_room():
    return [
        ('none', (0, 0, 0)),
        ('floor', (152, 223, 138)),
        ('ceiling', (78, 71, 183)),
        ('wall', (174, 199, 232)),
        ('door', (214, 39, 40)),
        ('window', (197, 176, 213)),
        ('light', (96, 207, 209)),
        ('prop', (31, 119, 180)),
        ('struct', (255, 187, 120))]


def create_color_palette_nyu():
    return [
        ('unlabeled', (0, 0, 0)),
        ('wall', (174, 199, 232)),
        ('floor', (152, 223, 138)),
        ('cabinet', (31, 119, 180)),
        ('bed', (255, 187, 120)),
        ('chair', (188, 189, 34)),
        ('sofa', (140, 86, 75)),
        ('table', (255, 152, 150)),
        ('door', (214, 39, 40)),
        ('window', (197, 176, 213)),
        ('bookshelf', (148, 103, 189)),
        ('picture', (196, 156, 148)),
        ('counter', (23, 190, 207)),
        ('blinds', (178, 76, 76)),
        ('desk', (247, 182, 210)),
        ('shelves', (66, 188, 102)),
        ('curtain', (219, 219, 141)),
        ('dresser', (140, 57, 197)),
        ('pillow', (202, 185, 52)),
        ('mirror', (51, 176, 203)),
        ('floormat', (200, 54, 131)),
        ('clothes', (92, 193, 61)),
        ('ceiling', (78, 71, 183)),
        ('books', (172, 114, 82)),
        ('refrigerator', (255, 127, 14)),
        ('television', (91, 163, 138)),
        ('paper', (153, 98, 156)),
        ('towel', (140, 153, 101)),
        ('showercurtain', (158, 218, 229)),
        ('box', (100, 125, 154)),
        ('whiteboard', (178, 127, 135)),
        ('person', (120, 185, 128)),
        ('nightstand', (146, 111, 194)),
        ('toilet', (44, 160, 44)),
        ('sink', (112, 128, 144)),
        ('lamp', (96, 207, 209)),
        ('bathtub', (227, 119, 194)),
        ('bag', (213, 92, 176)),
        ('otherstructure', (94, 106, 211)),
        ('otherfurniture', (82, 84, 163)),
        ('otherprop', (100, 85, 144))]


def create_color_palette_ade():
    return [
        ('wall', (120, 120, 120)),
        ('building, edifice', (180, 120, 120)),
        ('sky', (6, 230, 230)),
        ('floor, flooring', (80, 50, 50)),
        ('tree', (4, 200, 3)),
        ('ceiling', (120, 120, 80)),
        ('road, route', (140, 140, 140)),
        ('bed', (204, 5, 255)),
        ('windowpane, window', (230, 230, 230)),
        ('grass', (4, 250, 7)),
        ('cabinet', (224, 5, 255)),
        ('sidewalk, pavement', (235, 255, 7)),
        ('person, individual, someone, somebody, mortal, soul', (150, 5, 61)),
        ('earth, ground', (120, 120, 70)),
        ('door, double door', (8, 255, 51)),
        ('table', (255, 6, 82)),
        ('mountain, mount', (143, 255, 140)),
        ('plant, flora, plant life', (204, 255, 4)),
        ('curtain, drape, drapery, mantle, pall', (255, 51, 7)),
        ('chair', (204, 70, 3)),
        ('car, auto, automobile, machine, motorcar', (0, 102, 200)),
        ('water', (61, 230, 250)),
        ('painting, picture', (255, 6, 51)),
        ('sofa, couch, lounge', (11, 102, 255)),
        ('shelf', (255, 7, 71)),
        ('house', (255, 9, 224)),
        ('sea', (9, 7, 230)),
        ('mirror', (220, 220, 220)),
        ('rug, carpet, carpeting', (255, 9, 92)),
        ('field', (112, 9, 255)),
        ('armchair', (8, 255, 214)),
        ('seat', (7, 255, 224)),
        ('fence, fencing', (255, 184, 6)),
        ('desk', (10, 255, 71)),
        ('rock, stone', (255, 41, 10)),
        ('wardrobe, closet, press', (7, 255, 255)),
        ('lamp', (224, 255, 8)),
        ('bathtub, bathing tub, bath, tub', (102, 8, 255)),
        ('railing, rail', (255, 61, 6)),
        ('cushion', (255, 194, 7)),
        ('base, pedestal, stand', (255, 122, 8)),
        ('box', (0, 255, 20)),
        ('column, pillar', (255, 8, 41)),
        ('signboard, sign', (255, 5, 153)),
        ('chest of drawers, chest, bureau, dresser', (6, 51, 255)),
        ('counter', (235, 12, 255)),
        ('sand', (160, 150, 20)),
        ('sink', (0, 163, 255)),
        ('skyscraper', (140, 140, 140)),
        ('fireplace, hearth, open fireplace', (250, 10, 15)),
        ('refrigerator, icebox', (20, 255, 0)),
        ('grandstand, covered stand', (31, 255, 0)),
        ('path', (255, 31, 0)),
        ('stairs, steps', (255, 224, 0)),
        ('runway', (153, 255, 0)),
        ('case, display case, showcase, vitrine', (0, 0, 255)),
        ('pool table, billiard table, snooker table', (255, 71, 0)),
        ('pillow', (0, 235, 255)),
        ('screen door, screen', (0, 173, 255)),
        ('stairway, staircase', (31, 0, 255)),
        ('river', (11, 200, 200)),
        ('bridge, span', (255, 82, 0)),
        ('bookcase', (0, 255, 245)),
        ('blind, screen', (0, 61, 255)),
        ('coffee table, cocktail table', (0, 255, 112)),
        ('toilet, can, commode, crapper, pot, potty, stool, throne', (0, 255, 133)),
        ('flower', (255, 0, 0)),
        ('book', (255, 163, 0)),
        ('hill', (255, 102, 0)),
        ('bench', (194, 255, 0)),
        ('countertop', (0, 143, 255)),
        ('stove, kitchen stove, range, kitchen range, cooking stove', (51, 255, 0)),
        ('palm, palm tree', (0, 82, 255)),
        ('kitchen island', (0, 255, 41)),
        ('computer, computing machine, computing device, data processor, electronic computer, information processing system', (0, 255, 173)),
        ('swivel chair', (10, 0, 255)),
        ('boat', (173, 255, 0)),
        ('bar', (0, 255, 153)),
        ('arcade machine', (255, 92, 0)),
        ('hovel, hut, hutch, shack, shanty', (255, 0, 255)),
        ('bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', (255, 0, 245)),
        ('towel', (255, 0, 102)),
        ('light, light source', (255, 173, 0)),
        ('truck, motortruck', (255, 0, 20)),
        ('tower', (255, 184, 184)),
        ('chandelier, pendant, pendent', (0, 31, 255)),
        ('awning, sunshade, sunblind', (0, 255, 61)),
        ('streetlight, street lamp', (0, 71, 255)),
        ('booth, cubicle, stall, kiosk', (255, 0, 204)),
        ('television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', (0, 255, 194)),
        ('airplane, aeroplane, plane', (0, 255, 82)),
        ('dirt track', (0, 10, 255)),
        ('apparel, wearing apparel, dress, clothes', (0, 112, 255)),
        ('pole', (51, 0, 255)),
        ('land, ground, soil', (0, 194, 255)),
        ('bannister, banister, balustrade, balusters, handrail', (0, 122, 255)),
        ('escalator, moving staircase, moving stairway', (0, 255, 163)),
        ('ottoman, pouf, pouffe, puff, hassock', (255, 153, 0)),
        ('bottle', (0, 255, 10)),
        ('buffet, counter, sideboard', (255, 112, 0)),
        ('poster, posting, placard, notice, bill, card', (143, 255, 0)),
        ('stage', (82, 0, 255)),
        ('van', (163, 255, 0)),
        ('ship', (255, 235, 0)),
        ('fountain', (8, 184, 170)),
        ('conveyer belt, conveyor belt, conveyer, conveyor, transporter', (133, 0, 255)),
        ('canopy', (0, 255, 92)),
        ('washer, automatic washer, washing machine', (184, 0, 255)),
        ('plaything, toy', (255, 0, 31)),
        ('swimming pool, swimming bath, natatorium', (0, 184, 255)),
        ('stool', (0, 214, 255)),
        ('barrel, cask', (255, 0, 112)),
        ('basket, handbasket', (92, 255, 0)),
        ('waterfall, falls', (0, 224, 255)),
        ('tent, collapsible shelter', (112, 224, 255)),
        ('bag', (70, 184, 160)),
        ('minibike, motorbike', (163, 0, 255)),
        ('cradle', (153, 0, 255)),
        ('oven', (71, 255, 0)),
        ('ball', (255, 0, 163)),
        ('food, solid food', (255, 204, 0)),
        ('step, stair', (255, 0, 143)),
        ('tank, storage tank', (0, 255, 235)),
        ('trade name, brand name, brand, marque', (133, 255, 0)),
        ('microwave, microwave oven', (255, 0, 235)),
        ('pot, flowerpot', (245, 0, 255)),
        ('animal, animate being, beast, brute, creature, fauna', (255, 0, 122)),
        ('bicycle, bike, wheel, cycle', (255, 245, 0)),
        ('lake', (10, 190, 212)),
        ('dishwasher, dish washer, dishwashing machine', (214, 255, 0)),
        ('screen, silver screen, projection screen', (0, 204, 255)),
        ('blanket, cover', (20, 0, 255)),
        ('sculpture', (255, 255, 0)),
        ('hood, exhaust hood', (0, 153, 255)),
        ('sconce', (0, 41, 255)),
        ('vase', (0, 255, 204)),
        ('traffic light, traffic signal, stoplight', (41, 0, 255)),
        ('tray', (41, 255, 0)),
        ('ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', (173, 0, 255)),
        ('fan', (0, 245, 255)),
        ('pier, wharf, wharfage, dock', (71, 0, 255)),
        ('crt screen', (122, 0, 255)),
        ('plate', (0, 255, 184)),
        ('monitor, monitoring device', (0, 92, 255)),
        ('bulletin board, notice board', (184, 255, 0)),
        ('shower', (0, 133, 255)),
        ('radiator', (255, 214, 0)),
        ('glass, drinking glass', (25, 194, 194)),
        ('clock', (102, 255, 0)),
        ('flag', (92, 0, 255))]


def map_nyu2room():
    return [
        ('unlabeled', 'none'),
        ('wall', 'wall'),
        ('floor', 'floor'),
        ('cabinet', 'prop'),
        ('bed', 'prop'),
        ('chair', 'prop'),
        ('sofa', 'prop'),
        ('table', 'prop'),
        ('door', 'door'),
        ('window', 'window'),
        ('bookshelf', 'prop'),
        ('picture', 'prop'),
        ('counter', 'prop'),
        ('blinds', 'window'),
        ('desk', 'prop'),
        ('shelves', 'prop'),
        ('curtain', 'prop'),
        ('dresser', 'prop'),
        ('pillow', 'prop'),
        ('mirror', 'prop'),
        ('floormat', 'floor'),
        ('clothes', 'prop'),
        ('ceiling', 'ceiling'),
        ('books', 'prop'),
        ('refrigerator', 'prop'),
        ('television', 'prop'),
        ('paper', 'prop'),
        ('towel', 'prop'),
        ('showercurtain', 'prop'),
        ('box', 'prop'),
        ('whiteboard', 'prop'),
        ('person', 'window'),
        ('nightstand', 'prop'),
        ('toilet', 'struct'),
        ('sink', 'struct'),
        ('lamp', 'struct'),
        ('bathtub', 'struct'),
        ('bag', 'prop'),
        ('otherstructure', 'struct'),
        ('otherfurniture', 'prop'),
        ('otherprop', 'prop')]


def map_ade2nyu():
    return [('wall', 'wall'),
        ('building, edifice', 'window'),
        ('sky', 'window'),
        ('floor, flooring', 'floor'),
        ('tree', 'window'),
        ('ceiling', 'ceiling'),
        ('road, route', 'window'),
        ('bed', 'bed'),
        ('windowpane, window', 'window'),
        ('grass', 'window'),
        ('cabinet', 'cabinet'),
        ('sidewalk, pavement', 'window'),
        ('person, individual, someone, somebody, mortal, soul', 'person'),
        ('earth, ground', 'window'),
        ('door, double door', 'door'),
        ('table', 'table'),
        ('mountain, mount', 'window'),
        ('plant, flora, plant life', 'otherprop'),
        ('curtain, drape, drapery, mantle, pall', 'curtain'),
        ('chair', 'chair'),
        ('car, auto, automobile, machine, motorcar', 'window'),
        ('water', 'window'),
        ('painting, picture', 'picture'),
        ('sofa, couch, lounge', 'sofa'),
        ('shelf', 'shelves'),
        ('house', 'window'),
        ('sea', 'window'),
        ('mirror', 'mirror'),
        ('rug, carpet, carpeting', 'floormat'),
        ('field', 'window'),
        ('armchair', 'chair'),
        ('seat', 'chair'),
        ('fence, fencing', 'window'),
        ('desk', 'desk'),
        ('rock, stone', 'window'),
        ('wardrobe, closet, press', 'cabinet'),
        ('lamp', 'lamp'),
        ('bathtub, bathing tub, bath, tub', 'bathtub'),
        ('railing, rail', 'otherstructure'),
        ('cushion', 'pillow'),
        ('base, pedestal, stand', 'otherfurniture'),
        ('box', 'box'),
        ('column, pillar', 'otherstructure'),
        ('signboard, sign', 'whiteboard'),
        ('chest of drawers, chest, bureau, dresser', 'dresser'),
        ('counter', 'counter'),
        ('sand', 'window'),
        ('sink', 'sink'),
        ('skyscraper', 'window'),
        ('fireplace, hearth, open fireplace', 'wall'),
        ('refrigerator, icebox', 'refrigerator'),
        ('grandstand, covered stand', 'window'),
        ('path', 'window'),
        ('stairs, steps', 'otherstructure'),
        ('runway', 'window'),
        ('case, display case, showcase, vitrine', 'shelves'),
        ('pool table, billiard table, snooker table', 'table'),
        ('pillow', 'pillow'),
        ('screen door, screen', 'door'),
        ('stairway, staircase', 'otherstructure'),
        ('river', 'window'),
        ('bridge, span', 'otherstructure'),
        ('bookcase', 'bookshelf'),
        ('blind, screen', 'blinds'),
        ('coffee table, cocktail table', 'table'),
        ('toilet, can, commode, crapper, pot, potty, stool, throne', 'toilet'),
        ('flower', 'otherprop'),
        ('book', 'books'),
        ('hill', 'window'),
        ('bench', 'chair'),
        ('countertop', 'counter'),
        ('stove, kitchen stove, range, kitchen range, cooking stove', 'otherfurniture'),
        ('palm, palm tree', 'window'),
        ('kitchen island', 'counter'),
        ('computer, computing machine, computing device, data processor, electronic computer, information processing system', 'television'),
        ('swivel chair', 'chair'),
        ('boat','window'),
        ('bar', 'otherstructure'),
        ('arcade machine', 'otherfurniture'),
        ('hovel, hut, hutch, shack, shanty', 'window'),
        ('bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle', 'window'),
        ('towel', 'towel'),
        ('light, light source', 'lamp'),
        ('truck, motortruck', 'window'),
        ('tower', 'window'),
        ('chandelier, pendant, pendent', 'lamp'),
        ('awning, sunshade, sunblind', 'blinds'),
        ('streetlight, street lamp', 'window'),
        ('booth, cubicle, stall, kiosk', 'otherstructure'),
        ('television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box', 'television'),
        ('airplane, aeroplane, plane', 'window'),
        ('dirt track', 'window'),
        ('apparel, wearing apparel, dress, clothes', 'clothes'),
        ('pole', 'window'),
        ('land, ground, soil', 'window'),
        ('bannister, banister, balustrade, balusters, handrail', 'otherstructure'),
        ('escalator, moving staircase, moving stairway', 'otherstructure'),
        ('ottoman, pouf, pouffe, puff, hassock', 'chair'),
        ('bottle', 'otherprop'),
        ('buffet, counter, sideboard', 'counter'),
        ('poster, posting, placard, notice, bill, card', 'picture'),
        ('stage', 'otherstructure'),
        ('van', 'window'),
        ('ship', 'window'),
        ('fountain', 'window'),
        ('conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'otherfurniture'),
        ('canopy', 'window'),
        ('washer, automatic washer, washing machine', 'otherfurniture'),
        ('plaything, toy', 'otherprop'),
        ('swimming pool, swimming bath, natatorium', 'window'),
        ('stool', 'chair'),
        ('barrel, cask', 'otherprop'),
        ('basket, handbasket', 'otherprop'),
        ('waterfall, falls', 'window'),
        ('tent, collapsible shelter', 'otherprop'),
        ('bag', 'bag'),
        ('minibike, motorbike', 'window'),
        ('cradle', 'bed'),
        ('oven', 'otherprop'),
        ('ball', 'otherprop'),
        ('food, solid food', 'otherprop'),
        ('step, stair', 'otherstructure'),
        ('tank, storage tank', 'otherfurniture'),
        ('trade name, brand name, brand, marque', 'otherprop'),
        ('microwave, microwave oven', 'otherprop'),
        ('pot, flowerpot', 'otherprop'),
        ('animal, animate being, beast, brute, creature, fauna', 'window'),
        ('bicycle, bike, wheel, cycle', 'window'),
        ('lake', 'window'),
        ('dishwasher, dish washer, dishwashing machine', 'otherfurniture'),
        ('screen, silver screen, projection screen', 'television'),
        ('blanket, cover', 'otherprop'),
        ('sculpture', 'otherprop'),
        ('hood, exhaust hood', 'otherfurniture'),
        ('sconce', 'lamp'),
        ('vase', 'otherprop'),
        ('traffic light, traffic signal, stoplight', 'window'),
        ('tray', 'otherprop'),
        ('ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 'otherprop'),
        ('fan', 'lamp'),
        ('pier, wharf, wharfage, dock', 'window'),
        ('crt screen', 'television'),
        ('plate', 'otherprop'),
        ('monitor, monitoring device', 'television'),
        ('bulletin board, notice board', 'whiteboard'),
        ('shower', 'otherstructure'),
        ('radiator', 'otherfurniture'),
        ('glass, drinking glass', 'otherprop'),
        ('clock', 'otherprop'),
        ('flag', 'otherprop')]


def map_ade2room():
    nyu2room_list = map_nyu2room()
    nyu2room_dict = {}
    for k, v in nyu2room_list:
        nyu2room_dict[k] = v
    ade2nyu_list = map_ade2nyu()
    ade2room_list = []
    for k, v in ade2nyu_list:
        ade2room_list.append((k, nyu2room_dict[v]))
    return ade2room_list



if __name__ == '__main__':
    print(map_ade2room())

