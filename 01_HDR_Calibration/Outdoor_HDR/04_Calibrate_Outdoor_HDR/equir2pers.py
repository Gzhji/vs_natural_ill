import os
import cv2 
import lib.Equirec2Perspec as E2P
import lib.Perspec2Equirec as P2E
import lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse
import rawpy
import imageio
import matplotlib.pyplot as plt



def equir2pers(input_img):

    #
    # FOV unit is degree
    # theta is z-axis angle(right direction is positive, left direction is negative)
    # phi is y-axis angle(up direction positive, down direction negative)
    # height and width is output image dimension
    #
    equ = E2P.Equirectangular(input_img)    # Load equirectangular image
    img = equ.GetPerspective(60, 0, 0, 1024, 1024)  # Specify parameters(FOV, theta, phi, height, width)
    return img




if __name__ == '__main__':

    task = 'theta'

    if task == 'theta':
        input_img_directory = './theta_pano/'
        output_dir = './theta_pano/cropped/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        img_list = [a for a in os.listdir(input_img_directory) if a.endswith('.JPG')]
        for filename in img_list:
            rgb = cv2.imread(input_img_directory + filename)
            h, w = rgb.shape[:2]

            dim = (int(w*1.2), int(h*1.2))
            rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            img = rgb[int(0.4*h)+100:int(0.4*h) + 1300, int(0.45 * w) +200:int(0.45 * w) + 1700]

            output1 = output_dir + filename[:-4] + '.jpg'
            cv2.imwrite(output1, img)


    if task == '180':
        input_img_directory = './180_fisheye/'
        output_dir = './180_fisheye/cropped/'
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        img_list = [a for a in os.listdir(input_img_directory) if a.endswith('.JPG')]
        for filename in img_list:
            rgb = cv2.imread(input_img_directory + filename)
            h, w = rgb.shape[:2]
            img = rgb[int(0.35*h):int(0.35*h) + 1200, int(0.35 * w):int(0.35 * w) + 1500]
            output1 = output_dir + filename[:-4] + '.jpg'
            cv2.imwrite(output1, img)
