## Panoramic Furniture Detection
Original code, converting equirectangular panorama into 2D perspective view, is developed by Fu-En Wang.
[Original Code](https://github.com/fuenwang/Equirec2Perspec)

### 01 Perspective Segmentation from Panorama to 2D Perspective Images
run *python pano_furn_seg.py --task='equ2pers'*

The 2D perspective images will be saved into folder "1_Output_seg"

### 02 Semantic Segmentation for 2D Images
Implementing [Semantic Segmentation on MIT ADE20K dataset in PyTorch](https://github.com/CSAILVision/semantic-segmentation-pytorch) and run semantic segmentation for all images. 
Then, copy semantic images into folder "1_Output_seg_sem".

### 03 Binarization for Semantic Images
To highlight the furniture objects, run *python pano_furn_seg.py --task='sem2binary'*

The 2D perspective images will be saved into folder "1_Output_binary"

### 04 Stitching 2D Binarization Images to Panorama
To highlight the furniture objects, run *python pano_furn_seg.py --task='pers2equ'*

The panoramic furniture mask named '6_all_together.jpg' will be saved into folder "1_Output_seg".

