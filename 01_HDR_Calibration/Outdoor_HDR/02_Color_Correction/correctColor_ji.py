import argparse
from PIL import Image, ImageDraw, ImageFont
import csv
import cv2
import numpy as np
import glob
import os


def loadCCM(ccmCsvFile):
    with open(ccmCsvFile, encoding='utf-8') as f:
        csvData = f.read()

        lines = csvData.replace(' ', '').split('\n')
        del lines[len(lines) - 1]
        data = list()
        cells = list()

        for i in range(len(lines)):
            cells.append(lines[i].split(','))

        i = 0
        for line in cells:
            data.append(list())
            for j in range(len(line)):
                data[i].append(float(line[j]))
            i += 1

    return np.asarray(data)


def linear2SRGB(linear):
    linear = linear.copy()
    mask_low = linear <= 0.0031308
    linear[mask_low] *= 12.92
    linear[~mask_low] = (1.055 * linear[~mask_low]) ** (1 / 2.4) - 0.055
    linear = np.uint8(np.clip(linear, 0, 1.0) * 255.0)
    return linear


def SRGB2Linear(srgb):
    srgb = srgb.copy()
    mask_low = srgb <= 0.0404482
    srgb[mask_low] /= 12.92
    srgb[~mask_low] = ((srgb[~mask_low] + 0.055) / 1.055) ** 2.4  # linearized
    return srgb


def color_correction(sbgr_linear):
    """
    @param sbgr_linear: H x W x C
    @param M: 3 x 3 color correction matrix
    @return:
    """
    M = np.array([[0.412391, 0.357584, 0.180481],
                  [0.212639, 0.715169, 0.072192],
                  [0.019331, 0.119195, 0.950532]])

    # we want (H x W x C) @ (1, M^T) so it's a row vector times the M^T
    return sbgr_linear @ M.T[np.newaxis, ...]

def Apply_CCM(img, ccm):
    return img @ ccm[np.newaxis, ...]

def channelSplit(image):
    return np.dsplit(image, image.shape[-1])

def resize_hdr(img, target_w, target_h):
    dim = (target_w, target_h)
    new_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return new_img

def main(args):
    ccm = loadCCM(args.csv)
    print('ccm:', ccm)
    img = cv2.imread(args.input).astype('float32')
    img = img / 255.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    """
    apply CMM to LDR image to see the effect
    """
    input_img = SRGB2Linear(img)
    input_img = Apply_CCM(input_img, ccm)
    input_img = linear2SRGB(input_img)
    cv2.imwrite(args.output, cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR))

    """
    apply CMM to HDR image 
    """
    hdr_filter_img = cv2.imread(args.input_hdr, -1)
    hdr_filter_img = cv2.cvtColor(hdr_filter_img, cv2.COLOR_BGR2RGB)
    output_hdr = Apply_CCM(hdr_filter_img, ccm)
    output_hdr = np.float32(output_hdr)
    output_hdr = output_hdr * 1000 #ND filter reduced light admission exactly 1000 times
    cv2.imwrite(args.out_hdr, cv2.cvtColor(output_hdr, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apply CCM to images')
    parser.add_argument('--csv', type=str, default='data/CCM.csv', help='Path to the CCM file')
    parser.add_argument('--input', type=str, default='IMG_9850.JPG', help='Input image path')
    parser.add_argument('--output', type=str, default='ccm_filter_img.jpg', help='Output image path')
    parser.add_argument('--input_hdr', type=str, default='image_out_equ.hdr',  help='Input HDR image path')
    parser.add_argument('--out_hdr', type=str, default='image_out_equ_GTccm.hdr', help='Output HDR image path')

    args = parser.parse_args()
    main(args)