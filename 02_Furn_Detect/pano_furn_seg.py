import os
import cv2
import matplotlib.pyplot as plt
import lib.Equirec2Perspec as E2P
import lib.Perspec2Equirec as P2E
import lib.multi_Perspec2Equirec as m_P2E
import glob
import argparse
import numpy as np
import math
import sys


'''
Original code, which splits equirectangular panorama into 2D perspective view, is developed by Fu-En Wang.
Source: https://github.com/fuenwang/Equirec2Perspec
'''

def equir2pers(FOV = 60, theta_divid = 18, phi_divid = 8):
    """
    input:
        FOV: field of view for the cropped 2d perspective images
        theta_divid: divide the entire 360-degree azimuth angle into # divisions
        phi_divid: divide the entire 180-degree altitude angle into # divisions
    output:
        saved text file (named equ_list.txt) that describes images labeled theta and phi angles
    """

    if not os.path.exists(args.output_dir_seg):
        os.mkdir(args.output_dir_seg)
    print('input_img:', args.input_img)
    equ = E2P.Equirectangular(args.input_img)
    equ_list = []

    for i in range(theta_divid):
        for j in range(phi_divid):
            theta = i * 20
            phi = j * 6

            img = equ.GetPerspective(FOV, theta, -phi, 1024, 1024)
            output = os.path.join(args.output_dir_seg, 'theta_%03d_phi_%03d.jpg' % (theta, phi))
            cv2.imwrite(output, img)
            equ_list.append([FOV, theta, -phi])

    np.savetxt('equ_list.txt', equ_list, fmt='%1.4e')


def pers2equir(img_list, equ_list):
    """
    input:
        img_list: the annotated images
        equ_list: the theta and phi angles
    output:
        stitched panorama and mask
    """

    width = 2048
    height = 1024

    # this can turn cube to panorama
    equ = m_P2E.Perspective(img_list, equ_list)
    img = equ.GetEquirec(height, width)

    #get img mask
    mask = equ.GetMask(height, width)

    return img, mask

def alpha_blending(foreground,background,alpha):
    foreground = foreground.astype(float)
    background = background.astype(float)
    alpha = alpha.astype(float) / 255

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    outImage = cv2.add(foreground, background)

    return outImage

def resize(img, targ_height=1024):
    scale_percent = targ_height / img.shape[0]  # percent of original size's height
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return new_img

def DrawFrame(img):
    h, w = img.shape[:2]
    start_point = (0, 0)
    end_point = (w, h)
    color = (0, 255, 0)
    thickness = 10
    image = cv2.rectangle(img, start_point, end_point, color, thickness)
    return image

def OutLine(image, line_thick = 0, sort_threshold = 30):
    img = np.uint8(image)
    gray = DrawFrame(img)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours), "Contour objects.")
    sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)

    selected_cont = []
    for i in range(len(sorted_contour)):
        if len(sorted_contour[i]) > sort_threshold:
            selected_cont.append(sorted_contour[i])

    image_copy = gray.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, sorted_contour, 0, (0, 255, 0), thickness = line_thick)
    # print(len(selected_cont), "selected_cont objects.")

    h,w = gray.shape[:2]
    img_pl = np.zeros((h,w))

    cv2.fillPoly(img_pl, pts=selected_cont, color=(255, 255, 255))
    img_pl_con = cv2.drawContours(img_pl, selected_cont, -1, (255, 255, 255), 0)

    return img_pl_con, sorted_contour

def OutLine_furn(img, line_thick = 0, sort_threshold = 30):
    #img = np.uint8(image)
    gray = DrawFrame(img)
    #cv2.imwrite('img_gray.png', gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)
    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)
    selected_cont = []
    for i in range(len(sorted_contour)):
        if len(sorted_contour[i]) > sort_threshold:
            selected_cont.append(sorted_contour[i])

    return selected_cont

def Contour_Furn(image, flr_msk_file, line_thick = 0, sort_threshold = 10):
    img = np.uint8(image)
    gray = DrawFrame(img)
    #cv2.imwrite('img_gray.png', gray)

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 10, 100)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edged, kernel, iterations=1)

    contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contour = sorted(contours, key=cv2.contourArea, reverse=True)
    selected_cont = []
    for i in range(len(sorted_contour)):
        if len(sorted_contour[i]) > sort_threshold:
            selected_cont.append(sorted_contour[i])

    selected_cont = sorted_contour
    print(len(selected_cont), "selected_cont furn objects.")

    h,w = gray.shape[:2]
    img_pl = np.zeros((h,w))
    filtered_cont = []

    for i in range(len(selected_cont)):
        img_pl_copy = img_pl.copy()
        obj_msk = cv2.fillPoly(img_pl_copy, pts=selected_cont[i], color=255)
        obj_msk = cv2.drawContours(obj_msk, selected_cont[i], -1, 255, line_thick)
        flr_mask = cv2.imread(flr_msk_file, 0)
        flr_mask = resize(flr_mask, targ_height=obj_msk.shape[0])

        shift_dis = 10
        flr_mask = np.roll(flr_mask, - shift_dis, axis=0)
        flr_mask = cv2.rectangle(flr_mask, (0, flr_mask.shape[0]-shift_dis), (flr_mask.shape[1], flr_mask.shape[0]), 255, -1)

        if np.sum(flr_mask[obj_msk == 255]) != 0:
            filtered_cont.append(selected_cont[i])

    img_canvas = np.zeros((h,w))
    cv2.fillPoly(img_canvas, pts=filtered_cont, color=(255, 255, 255))
    img_pl_con = cv2.drawContours(img_canvas, filtered_cont, -1, (255, 255, 255), line_thick)

    return img_pl_con

def Get_Furn(img):
    img = np.where(img == 106, 0, img) # exclude curtain
    img = np.where(img == 106, 255, img) #table
    img = np.where(img == 89, 255, img)  #table
    img = np.where(img == 176, 255, img) #armchair
    img = np.where(img == 92, 255, img) #sofa;couch;lounge
    img = np.where(img == 93, 255, img)  #bed
    img = np.where(img == 160, 255, img) #desk
    img = np.where(img == 167, 255, img)  #pillow
    img = np.where(img == 190, 255, img) #chair
    img = np.where(img == 102, 255, img) #chair
    #img = np.where(img == 64, 255, img) #bathtub
    #img = np.where(img == 171, 255, img)  # tv
    img = np.where(img == 211, 255, img)  # indoor plant
    img = np.where(img == 217, 255, img)  # lamp
    img = np.where(img == 88, 255, img) #shelf
    img = np.where(img == 85, 255, img)  #painting/pic
    img = np.where(img == 79, 255, img)    #plaything; toy
    img = np.where(img == 147, 255, img)  #bag
    img = np.where(img == 98, 255, img)# trash;can;

    img = np.where(img == 255, 255, 0)

    return img

def Get_Centroid(Contour_List_Arr):
    x_pts = (Contour_List_Arr[:, 0])[:, 0]  # need [:, 0] decrease a dimension from contour
    y_pts = (Contour_List_Arr[:, 0])[:, 1]
    win_center_pt = (int(np.average(x_pts)), int(np.average(y_pts)))
    return win_center_pt

def To_Left(data, dis = 10):
    left = np.roll(data, - dis, axis=1)
    left[:,0] = True
    return left

def To_Right(data, dis = 10):
    right = np.roll(data, dis, axis=1)
    right[:,-1] = True
    return right

def To_Down(data, dis = 10):
    down = np.roll(data, dis, axis=0)
    down[-1, :] = True
    return down

def Get_Slope(pt1, pt2):
    x1 = pt1[0]
    y1 = pt1[1]
    x2 = pt2[0]
    y2 = pt2[1]
    return (y2-y1)/(x2-x1)

def Resize(img, target_h):
    dim = (target_h*2, target_h)
    new_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return new_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='panoramic furniture segmentation')
    parser.add_argument('--task', type=str, default='tripod_paint',
                        choices=['equ2pers', 'sem2binary', 'pers2equ', 'tripod_paint', 'detect_shadow'],
                        help='The task to perform')
    parser.add_argument('--FOV', type=int, default=90, help='Field of View (FOV) as an integer')
    parser.add_argument('--theta_divd', type=int, default=20, help='number of division for theta as an integer')
    parser.add_argument('--phi_divd', type=int, default=10, help='number of division for phi as an integer')
    parser.add_argument('--input_img', type=str, default='0_Panorama/new+penn+bed+07091127.jpg', help='Path to the input panoramic image')
    parser.add_argument('--tripod_file', type=str, default='0_Tripod/new+penn+bed+07091127_tripod.png', help='Path to the panoramic tripod file')
    parser.add_argument('--sem_file', type=str, default= '0_Semantic/new+penn+bed+07091127_sem.png', help='Path to the panoramic semantic file')
    parser.add_argument('--flr_msk_file', type=str, default='0_Floor_Boundary/new+penn+bed+07091127.png', help='Path to the floor mask file')
    parser.add_argument('--sunlit_file', type=str, default= '0_Sunlight/new+penn+bed+07091127_light.png', help='Path to the sunlit file')
    parser.add_argument('--output_dir_seg', type=str, default= '1_Output_seg/', help='Path to the output directory')
    parser.add_argument('--output_dir_semantic', type=str, default= '1_Output_seg_sem/', help='Path to the output directory for semantic images')
    parser.add_argument('--output_dir_binary', type=str, default= '1_Output_binary/', help='Path to the output directory for binary images')
    parser.add_argument('--pixel_thred', type=int, default=5, help='threshold for pixel classified into furniture')
    parser.add_argument('--result_path', type=str, default='2_Results/', help='Path to the results')
    parser.add_argument('--shift_y', type=float, default= 5, help='switch distance in y direction')
    parser.add_argument('--ratio_cof', type=float, default=0.5, help='slope for switch distance')

    # Parse the arguments
    args = parser.parse_args()

    if args.task == 'equ2pers':
        # segment the input_img into multiple 2d perspective images
        equir2pers(FOV = args.FOV, theta_divid = args.theta_divd, phi_divid = args.phi_divd)
        print('perspective done')

    elif args.task == 'sem2binary':
        # highlight target furniture objects from 2D semantic sperspective
        img_list = []
        saved_equ = np.loadtxt('equ_list.txt', dtype='float')
        filename_list = [a for a in os.listdir(args.output_dir_semantic)]
        filename_list.sort()

        for filename in filename_list:
            img = cv2.imread(args.output_dir_semantic + filename, 0)
            img = Get_Furn(img)
            cv2.imwrite(args.output_dir_binary + filename, img)
        print('sem2binary done')

    elif args.task == 'pers2equ':
        img_list = []
        saved_equ = np.loadtxt('equ_list.txt', dtype='float')
        filename_list = [a for a in os.listdir(args.output_dir_binary)]
        filename_list.sort()

        for filename in filename_list:
            splits = filename[:-4].split('_')
            theta, phi = splits[1], splits[3]
            print('theta:', theta, 'phi:', phi)
            img_list.append(args.output_dir_binary + filename)

        #stitch all imgs together
        stitch_img, mask = pers2equir(img_list, saved_equ)
        cv2.imwrite(args.result_path + '0_output.png', stitch_img)

        #if pixel classified than threshold times, then include it in the result'
        stitch_img[stitch_img <= args.pixel_thred] = 0
        stitch_img = stitch_img.astype(bool)
        stitch_img = stitch_img.astype(int) * 255
        cv2.imwrite(args.result_path + '0_output_thred.png', stitch_img)

        mask = mask.astype(int) * 255
        sem_layer = cv2.imread(args.sem_file, 0)
        mask[sem_layer == 230] = 0
        cv2.imwrite(args.result_path + '1_binary_mask.png', mask)

        h, _ = mask.shape[:2]
        background = cv2.imread(args.input_img)
        background = resize(background, targ_height=h)

        win_color = 230
        pano_sem = cv2.imread(args.sem_file, 0)
        mask[args.sem_file == 230] = 0

        org_mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
        dst_org = (background + org_mask)/2
        cv2.imwrite(args.result_path + '3_result_org_mask.png', dst_org)
        print('img2pano done')
    #
    #
    elif args.task == 'tripod_paint':
        tripod_layer = cv2.imread(args.tripod_file, 0)
        furn_mask = cv2.imread(args.result_path + '0_output_thred.png', 0)

        # #location of checker board
        # empty = np.zeros(furn_mask.shape[:2])
        # empty = cv2.rectangle(empty, (846, 733), (990, 833), 255, -1)
        # furn_mask[empty == 255] = 0

        img_pl_con = Contour_Furn(furn_mask, args.flr_msk_file, line_thick = 30, sort_threshold = 10)

        tripod_layer = Resize(tripod_layer, furn_mask.shape[0])
        tripod_layer[tripod_layer != 0] = 255

        final_img = img_pl_con + tripod_layer
        cv2.imwrite(args.result_path + '4_img_pl_conmask.jpg', final_img)

        ori_img = cv2.imread(args.input_img)
        img_pl_con = np.repeat(final_img[..., np.newaxis], 3, axis=2)
        final_img = img_pl_con/2 + ori_img/2
        cv2.imwrite(args.result_path + '4_final_img.jpg', final_img)
        print('tripod_paint done')


    elif args.task == 'detect_shadow':

        win_color = 230 #glass
        floor_color = 58

        orig_furn = cv2.imread(args.result_path + '4_img_pl_conmask.jpg', 0)
        pano_sem = cv2.imread(args.sem_file, 0)
        floor_edge = cv2.imread(args.flr_msk_file, 0)

        tripod_layer = cv2.imread(args.tripod_file, 0)
        sun_layer = cv2.imread(args.sunlit_file, 0)
        sun_layer[sun_layer != 0] = 255
        win_sem = np.where(pano_sem == win_color, 255, 0)

        _, win_list = OutLine(win_sem, line_thick=0, sort_threshold=10)
        win_list_arr = np.array(win_list)

        win_center_pt = Get_Centroid(win_list_arr[0])
        print('win_center:', win_center_pt)

        'get furn mask points'
        furn_list = OutLine_furn(orig_furn, line_thick=0, sort_threshold=0)
        furn_list_arr = np.array(furn_list)
        furn_cen_list = []
        for i in range(len(furn_list)):
            furn_center_pt = Get_Centroid(furn_list_arr[i])
            furn_cen_list.append(furn_center_pt)


        shadow_all = np.zeros(orig_furn.shape[:2])
        for i in range(len(furn_cen_list)):
            pt_slope = Get_Slope(win_center_pt, furn_cen_list[i]) * args.ratio_cof
            shift_x = int(args.shift_y/pt_slope)
            canvas = np.zeros(orig_furn.shape[:2])
            furn_img = cv2.drawContours(canvas, furn_list, i, (255, 255, 255), -1)
            furn_img = cv2.drawContours(furn_img, furn_list, i, (255, 255, 255), 40)
            if shift_x > 0:
                furn_img = To_Right(furn_img, dis = shift_x)
            else:
                furn_img = To_Left(furn_img, dis = -shift_x)
            furn_img = To_Down(furn_img, dis = args.shift_y)
            shadow_all += furn_img

        canvas_flr = np.zeros(orig_furn.shape[:2])
        furn2floor = cv2.drawContours(canvas_flr, furn_list, -1, (255, 255, 255), -1)

        if furn2floor.shape[0] != floor_edge.shape[0]:
            furn2floor = Resize(furn2floor, floor_edge.shape[0])
        furn2floor[floor_edge != 255] = 0

        orig_furn = Resize(orig_furn, floor_edge.shape[0])
        shadow_all = Resize(shadow_all, floor_edge.shape[0])
        sun_layer = Resize(sun_layer, floor_edge.shape[0])
        tripod_layer = Resize(tripod_layer, floor_edge.shape[0])
        all_shadow = orig_furn + shadow_all + furn2floor + sun_layer + tripod_layer

        sem_layer = cv2.imread(args.sem_file, 0)
        sem_layer = resize(sem_layer, targ_height=all_shadow.shape[0])
        all_shadow[sem_layer == 230] = 0
        all_shadow = np.repeat(all_shadow[..., np.newaxis], 3, axis=2)
        cv2.imwrite(args.result_path + '6_all_together.jpg', all_shadow)

        background = cv2.imread(args.input_img)
        background = Resize(background, floor_edge.shape[0])
        overlap_shadow = all_shadow/2 + background/2
        cv2.imwrite(args.result_path + '5_overlap_shadow.png', overlap_shadow)
        print('detect_shadow done')

    else:
        pass