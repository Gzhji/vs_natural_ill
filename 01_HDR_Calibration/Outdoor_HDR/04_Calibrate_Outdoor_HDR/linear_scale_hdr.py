import cv2
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process Outdoor HDR image.")
    parser.add_argument("--input_hdr", type=str, default='image_out_equ_GTccm.hdr', required=True, help="Input HDR file path")
    parser.add_argument("--output_hdr", type=str, default='image_out_equ_GTccm_cali.hdr', required=True, help="Output HDR file path")
    parser.add_argument("--indoor_displayed_L", type=float, default=1019, required=True, help="Indoor displayed luminance")
    parser.add_argument("--indoor_measured_L", type=float, default=34.63, required=True, help="Indoor measured luminance")
    parser.add_argument("--theta2fisheye", type=float, default=2.117, required=True, help="Theta to fisheye value")

    return parser.parse_args()


def main():
    args = parse_args()

    # Constants
    indoor_displayed_L = args.indoor_displayed_L
    indoor_measured_L = args.indoor_measured_L
    theta2fisheye = args.theta2fisheye

    # Calculate scene_k
    scene_k = indoor_displayed_L / indoor_measured_L
    print('scene_k1:', scene_k)

    # Load the HDR image
    hdr_filter_img = cv2.imread(args.input_hdr, -1)
    print('Original max:', np.max(hdr_filter_img))

    # Apply transformations
    hdr_cali_img = hdr_filter_img / scene_k
    hdr_cali_img = hdr_cali_img / theta2fisheye
    print('Calibrated max:', np.max(hdr_cali_img))

    # Save the output HDR image
    cv2.imwrite(args.output_hdr, hdr_cali_img)

if __name__ == '__main__':
    main()

