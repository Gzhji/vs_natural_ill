# Indoor - Outdoor HDR Calibration
This section introduces details for calibrating indoor and outdoor HDR images.


## Indoor HDR Calibration
The photometric calibration for indoor HDR panorama follows the per-pixel calibration in [hdrscope](https://courses.washington.edu/hdrscope/index.html).
The measured absolute luminance will be used to linearly scale each HDR image.


## Outdoor HDR Calibration
The photometric calibration for outdoor HDR panorama follows a series of steps.

### Vignetting Correction
The captured 180-degree fisheye () will be paired with  


Using radiance command: *pcomb -e "ro=ri(1)/ri(2);go=gi(1)/gi(2);bo=bi(1)/bi(2)" image_in.hdr mask.hdr > image_out.hdr* 
