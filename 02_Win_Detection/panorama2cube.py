import os
import cv2 
import lib.Equirec2Perspec as E2P
import lib.Perspec2Equirec as P2E
import lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse
import numpy as np
from PIL import Image, ImageDraw


def panorama2cube(input_dir,output_dir):

    cube_size = 640
    cube_h = 640
    cube_w = 640

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    all_image = sorted(glob.glob(input_dir + '/*.*'))
    print(all_image)

    for index in range(len(all_image)):
        # image = '../Opensfm/source/library/test-1/frame{:d}.png'.format(i)
        equ = E2P.Equirectangular(all_image[index])    # Load equirectangular image
        #
        # FOV unit is degree
        # theta is z-axis angle(right direction is positive, left direction is negative)
        # phi is y-axis angle(up direction positive, down direction negative)
        # height and width is output image dimension
        #

        out_dir = output_dir + '/%02d/'%(index)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        img = equ.GetPerspective(90, 0, 0, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output1 = out_dir + 'front.jpg'
        cv2.imwrite(output1, img)
        output11 = out_dir + 'front.png'
        cv2.imwrite(output11, img)

        img = equ.GetPerspective(90, 90, 0, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output2 = out_dir + 'right.png' 
        cv2.imwrite(output2, img)

        img = equ.GetPerspective(90, 180, 0, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output3 = out_dir + 'back.png' 
        cv2.imwrite(output3, img)

        img = equ.GetPerspective(90, 270, 0, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output4 = out_dir + 'left.png' 
        cv2.imwrite(output4, img)

        img = equ.GetPerspective(90, 0, 90, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output5 = out_dir + 'top.png' 
        cv2.imwrite(output5, img)

        img = equ.GetPerspective(90, 0, -90, cube_h, cube_w)  # Specify parameters(FOV, theta, phi, height, width)
        output6 = out_dir + 'bottom.png' 
        cv2.imwrite(output6, img)


def cube2panorama(input_dir, output_dir):

    width = 2048
    height = 1024

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    front = input_dir + '/front.png'
    right = input_dir + '/right.png'
    back = input_dir + '/back.png'
    left = input_dir + '/left.png'
    top = input_dir + '/top.png'
    bottom = input_dir + '/bottom.png'

    # this can turn cube to panorama
    per = m_P2E.Perspective([front,right,back,left,top,bottom],
                            [[90, 0, 0],[90, 90, 0],[90, 180, 0],
                            [90, 270, 0],[90, 0, 90],[90, 0, -90]])


    img = per.GetEquirec(height,width)
    img[img != 255] = 0
    cv2.imwrite(output_dir + '/cube_output.png', img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='cube', choices=['panorama', 'cube', 'ref_cube', 'fisheye'])
    parser.add_argument('--input', type=str, default='./panorama')
    parser.add_argument('--output', type=str, default='./pano_output')
    config = parser.parse_args()

    if config.mode == 'panorama':
        panorama2cube(config.input, config.output)

    elif config.mode == 'cube':
        win_pattern_path = '01_frame.png'

        front = cv2.imread(win_pattern_path, 0)
        cv2.imwrite(config.output + '/front.png', front)

        h, w = front.shape[:2]
        empty_mask = np.zeros((h, w, 3))

        cv2.imwrite(config.output + '/right.png', empty_mask)
        cv2.imwrite(config.output + '/back.png', empty_mask)
        cv2.imwrite(config.output + '/left.png', empty_mask)
        cv2.imwrite(config.output + '/top.png', empty_mask)
        cv2.imwrite(config.output + '/bottom.png', empty_mask)
        cube2panorama(config.output, './output')

