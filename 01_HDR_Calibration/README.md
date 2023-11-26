# Indoor - Outdoor HDR Calibration
This section introduces details for calibrating indoor and outdoor HDR images.
Please install [Radiance](https://github.com/LBNL-ETA/Radiance/releases) for this part. 


## Indoor HDR Calibration
The photometric calibration for indoor HDR panorama follows the per-pixel calibration in [hdrscope](https://courses.washington.edu/hdrscope/index.html).
The measured absolute luminance will be used to linearly scale each HDR image.


## Outdoor HDR Calibration
The photometric calibration for outdoor HDR panorama follows a series of steps.

### 01 Vignetting Correction
The captured 180-degree fisheye image (named as *image_in.hdr*) will be paired with *mask.hdr* to compensate for the light loss at the periphery area. 

Using radiance command: *pcomb -e "ro=ri(1)/ri(2);go=gi(1)/gi(2);bo=bi(1)/bi(2)" image_in.hdr mask.hdr > image_out.hdr* 

The *image_out.hdr* needs to be cropped into square image. 

### 02 Equidistant Correction
To transform the picture from angular fish eye to hemispherical fish eye, Using radiance command:
*pinterp -vf image_out.hdr -vth –x (aliasx) –y (aliasy) -ff image_out.hdr 1 > image_out_equ.hdr*

### 03 Color Correction
Copy *image_out_equ.hdr* into '03_Color_Correction' folder, and run python file *correct_Color.py*.

The export file after color correction is *image_out_equ_GTccm.hdr*.

### 04 Calibrate Outdoor HDR
Copy the *'image_out_equ_GTccm.hdr'* into '04_Calibrate_Outdoor_HDR' folder, and run python file *linear_scale_hdr.py*.
This step requires *indoor_displayed_L* (the value displayed in the original HDR before calirbation) and *indoor_measured_L* (absolute value measured by luminance meter) from indoor panorama.

The export file is *image_out_equ_GTccm_cali.hdr*
