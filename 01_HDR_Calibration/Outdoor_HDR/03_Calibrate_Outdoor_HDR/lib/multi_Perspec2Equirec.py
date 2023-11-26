import os
import sys
import cv2
import numpy as np
import lib.Perspec2Equirec as P2E


class Perspective:
    def __init__(self, img_array , F_T_P_array ):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

    def GetEquirec(self, height, width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, 3))
        merge_mask = np.zeros((height, width, 3))

        for img_dir, [F, T, P] in zip(self.img_array, self.F_T_P_array):
            per = P2E.Perspective(img_dir, F, T, P)  # Load equirectangular image
            img, mask = per.GetEquirec(height, width)  # Specify parameters(FOV, theta, phi, height, width)
            merge_image += img
            merge_mask += mask

        merge_mask = np.where(merge_mask == 0, 1, merge_mask)
        merge_image = (np.divide(merge_image, merge_mask))

        return merge_image


    def GetMask(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3), dtype=bool)
        merge_mask = np.zeros((height,width,), dtype=bool)

        from matplotlib import pyplot as plt

        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = P2E.Perspective(img_dir,F,T,P)        # Load equirectangular image
            img, mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            # print('max, min pixel:', np.max(img), np.min(img))

            # print(np.unique(img))
            # plt.imshow(img.astype(int))
            # plt.show()
            furniture_mask = (img[:, :, 0].squeeze() == 255)
            merge_mask = merge_mask | furniture_mask
            #plt.imshow(merge_mask.astype(int))
            # plt.show()
            # merge_image += img
            #merge_mask = merge_mask | mask

        # merge_mask = np.where(merge_mask==0,1,merge_mask)
        #merge_image = (np.divide(merge_image,merge_mask))
        # print('after:', np.max(merge_image), np.min(merge_image))

        # plt.imshow(merge_mask.astype(int))
        # plt.show()

        return merge_mask
        






