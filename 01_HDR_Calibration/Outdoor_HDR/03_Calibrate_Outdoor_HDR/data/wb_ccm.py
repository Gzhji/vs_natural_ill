from email.mime import image
from genericpath import exists
from platform import architecture
from turtle import color
from unittest.mock import patch
import numpy as np
import cv2
from skimage.io import imread
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import scipy
import json
from cp_hw2 import writeHDR, readHDR, read_colorchecker_gm

def read_color_chart(img_path: str, chart_pxy_path: str = "./chart_pxy.json"):
    with open(chart_pxy_path, "r") as f:
        chart_pxy = json.load(f)
    # [24, 2]
        chart_pxy = np.array(chart_pxy).astype(np.int32)
    # print("debug", chart_pxy.shape)

    img = readHDR(img_path)
    color_list = []

    x_width = 5
    y_windth = 10
    for pxy in chart_pxy:

        patch = img[
            pxy[1] - y_windth : pxy[1] + y_windth,
            pxy[0] - x_width : pxy[0] + x_width,
            :,
        ].reshape(-1, 3)

        color_list.append(np.median(patch, axis=0))

    color_array = np.stack(color_list, axis=0)  # .reshape(4, 6, 3)

    # [4 x 6, 3]
    return color_array

def get_affine_lest_sq(x, y):
    """
    x: shape [N, 3]
    y: shape [N, 3]
    """

    x_homo = np.concatenate([x, np.ones((x.shape[0], 1))], axis=-1)

    M, residual, rank, s = scipy.linalg.lstsq(x_homo, y)

    print("affine error: ", residual)
    return M.transpose()

def color_transform(
    img_path: str,
    chart_pxy_path: str = "./chart_pxy.json", save_path: str = "./tmp.hdr"):

    white_ind = 16
    # [4 x 6, 3]
    color_array = read_color_chart(img_path, chart_pxy_path)
    # [4, 6]
    gt_r, gt_g, gt_b = read_colorchecker_gm()

    # [24, 3]
    gt_color = np.stack([gt_r, gt_g, gt_b], axis=-1).reshape(-1, 3)

    # M is [3, 4] => [A, b]
    # val, M, inliners = cv2.estimateAffine3D(color_array, gt_color)
    M = get_affine_lest_sq(color_array, gt_color)
    error = np.matmul(
        M, np.concatenate([color_array, np.ones((24, 1))], axis=-1).transpose()
    ).transpose()
    error = np.abs(error - gt_color)
    error_mean = np.mean(error)

    wb_scaling = gt_color[white_ind] / color_array[white_ind]
    # print(gt_color[white_ind], color_array[white_ind])
    # print("homo", val, M, error_mean)
    # print(error)
    # print(gt_color)

    # [H, W, 3]
    source_img = readHDR(img_path)
    img_size = source_img.shape

    # [4, H, W]
    homo_img = np.concatenate(
        [source_img, np.ones([*source_img.shape[:2], 1])], axis=-1
    ).transpose(2, 0, 1)

    # print(homo_img.shape, M.shape)
    transformed_image = np.matmul(M, homo_img.reshape(4, -1)).reshape(3, *img_size[:2])

    # print("debug", transformed_image.shape)

    ret_img = transformed_image.transpose(1, 2, 0)

    #
        with open(chart_pxy_path, "r") as f:
            chart_pxy = json.load(f)[white_ind]

            chart_pxy = [int(_) for _ in chart_pxy]
            white_patch = (
                ret_img[
                    chart_pxy[1] - 10 : chart_pxy[1] + 10,
                    chart_pxy[0] - 5 : chart_pxy[0] + 5,
                    :,
                ]
                .mean(axis=0)
                .mean(axis=0)
            )

    wb_scaling = gt_color[white_ind] / white_patch
    ret_img = ret_img * wb_scaling[np.newaxis, np.newaxis, :]
    print("debug whitebalance: ", wb_scaling)
    ret_img = np.clip(ret_img, a_min=0.0, a_max=None)
    print(ret_img.shape)

    writeHDR(save_path, ret_img)

if __name__ == "__main__":
    color_transform(img_path="../data/door_stack_results/linear_raw_gaussian.hdr")