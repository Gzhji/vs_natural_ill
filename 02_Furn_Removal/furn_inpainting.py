import cv2
import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np

def Flip_Img(img):
    """
    Flips the input image horizontally.
    """
    h, w = img.shape[:2]
    l_img = img[0:h, 0:w//2]
    r_img = img[0:h, w//2:w]
    return np.concatenate((r_img, l_img), axis=1)


def Resize_Img(img, targ_width = 2048, targ_hight = 1024):
    dim = (targ_width, targ_hight)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def Mask3Channel(mask):
    return np.repeat(mask[:, :, np.newaxis], 3, axis=2)

def Get_flr_Masks(scene_img, flor_msk, furn_msk):
    """
    Perform a series of image operations involving masks and textures.
    Args:
        input_image (numpy.ndarray): Scene image.
        floor_mask (numpy.ndarray): Floor mask.
        furniture_mask (numpy.ndarray): Furniture mask.

    """
    unknown_msk = furn_msk.copy()
    revs_mask = ~unknown_msk.astype(bool)
    revs_mask = revs_mask.astype(int) * 255
    revs_mask[flor_msk == 0] = 0
    # cv2.imwrite(opt.log + '4_all_mask_f.png', revs_mask)

    flor_tex = scene_img.copy()
    flor_tex[revs_mask == 0] = np.array([255, 255, 255])
    cv2.imwrite(opt.log + '4_flor_tex.png', flor_tex)

    lama_furn_msk = furn_msk.copy()
    lama_furn_msk[flor_msk == 0] = 255
    cv2.imwrite(opt.log + '4_flor_tex_mask.png', lama_furn_msk)

    # valid_mask =  np.ones(scene_img.shape) *255
    # valid_mask[flor_tex == np.array([255,255,255])] = 0
    # cv2.imwrite(opt.log + '4_flor_valid_mask.png', valid_mask)


def Get_Wall_Masks(scene_img, non_flor_msk, furn_msk):
    """
    Perform a series of image operations involving masks and textures for walls.
    Args:
        input_image (numpy.ndarray): Scene image.
        non_floor_mask (numpy.ndarray): Non-floor mask.
        furniture_mask (numpy.ndarray): Furniture mask.

    """
    unknown_msk = furn_msk.copy()
    revs_mask = ~unknown_msk.astype(bool)
    revs_mask = revs_mask.astype(int) * 255
    revs_mask[non_flor_msk == 0] = 0
    # cv2.imwrite(opt.log + '4_all_mask_w.png', revs_mask)

    flor_tex = scene_img.copy()
    flor_tex[revs_mask == 0] = np.array([255, 255, 255])
    cv2.imwrite(opt.log + '4_wall_tex.png', flor_tex)

    lama_furn_msk = furn_msk.copy()
    lama_furn_msk[non_flor_msk == 0] = 255
    cv2.imwrite(opt.log + '4_wall_tex_mask.png', lama_furn_msk)


def Resize(img, target_h):
    dim = (target_h*2, target_h)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--house', type=str, default='penn')
    parser.add_argument('--floor', type=str, default='bed')
    parser.add_argument('--pano', type=str, default='07091127')
    parser.add_argument('--log', type=str, default='../03_Reflectance_Tex/out_obj')
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--task', type=str, default='concatenate') # task = 'concatenate'

    return parser.parse_args()


if __name__ == '__main__':

    opt = parse_args()
    opt.log = f'{opt.log}/{opt.house}+{opt.floor}+{opt.pano}/'
    print('path to the scene folder is:', opt.log)

    """
    read normal map and scene image
    """
    scene_img = cv2.imread(opt.log + '0_rgb_i.png')
    normal_map = cv2.imread(opt.log + 'normal.png', 0)
    h = opt.height
    w = h * 2

    scene_img = Resize_Img(scene_img, targ_width=w, targ_hight=h)
    normal_map = Resize_Img(normal_map, targ_width=w, targ_hight=h)

    """
    save floor and non floor masks
    """
    canvas = np.zeros((h,w))
    flr_msk = canvas.copy()
    flr_msk[normal_map == 95] = 255
    cv2.imwrite(opt.log + 'floor_mask.png', flr_msk)

    non_flr_msk = canvas.copy()
    non_flr_msk[normal_map != 95] = 255
    cv2.imwrite(opt.log + 'non_floor_mask.png', non_flr_msk)

    """
    combine furn mask with flor mask
    """
    furn_msk = cv2.imread(opt.log + '6_all_together.jpg', 0)
    furn_msk[furn_msk != 255] = 0

    if opt.task == 'mask_prep':

        inverse_fur_msk = furn_msk.copy()
        inverse_fur_msk = ~inverse_fur_msk.astype(bool)
        inverse_fur_msk = inverse_fur_msk.astype(int) * 255
        cv2.imwrite(opt.log + '4_inverse_fur_msk.png', inverse_fur_msk)

        masked_scene = scene_img.copy()
        masked_scene[inverse_fur_msk == 0] = np.array([255,255,255])
        cv2.imwrite(opt.log + '4_masked_scene.png', masked_scene)

        Get_flr_Masks(scene_img, flr_msk, furn_msk)
        Get_Wall_Masks(scene_img, non_flr_msk, furn_msk)
    #
    #
    #
    if opt.task == 'concatenate':

        Pred_Floor_path = opt.log + '4_flor_tex_mask_lama.png'
        Pred_Floor = cv2.imread(Pred_Floor_path)
        Pred_Floor = Resize_Img(Pred_Floor , targ_width = w, targ_hight = h)
        Pred_Floor[flr_msk == 0] = np.array([0,0,0])

        Pred_Wall_path = opt.log + '4_wall_tex_mask_lama.png'
        Pred_Wall = cv2.imread(Pred_Wall_path)
        Pred_Wall = Resize_Img(Pred_Wall , targ_width = w, targ_hight = h)
        Pred_Wall[non_flr_msk == 0] = np.array([0,0,0])

        complete_img = Pred_Wall + Pred_Floor
        cv2.imwrite(opt.log + 'complete_Pred_Wall.jpg', Pred_Wall)
        cv2.imwrite(opt.log + 'complete_Pred_Floor.jpg', Pred_Floor)
        cv2.imwrite(opt.log + 'complete_img.jpg', complete_img)