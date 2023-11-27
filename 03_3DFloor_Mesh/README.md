## Panoramic Furniture Detection
Please implement [semantic segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch) and [line detection](https://github.com/zhou13/lcnn). 

### 01 Data Preparation 
Place a panorama image under folder 'panorama', and run *panorama2cube.py --task='panorama'*

The cubic images are saved into folder 'pano_output'.

Using the front.jpg (.png) for [window segmentation](https://github.com/CSAILVision/semantic-segmentation-pytorch) for window area and [line detection](https://github.com/zhou13/lcnn) for window frame. 

Place the output images from two models into folder 'img'.

### 02 Generate Window Mask
run *win_frame.py --sem_path='img/front.png' --line_path='img/front-0.84.png' --process_mode='cube'* 

The export image is '01_frame.png'.

Copy '01_frame.png' into folder 'pano_output' and rename it as 'front.png'

Run *panorama2cube.py --task='cube'*

The export image is under folder 'output' and named 'cube_output.png'.
