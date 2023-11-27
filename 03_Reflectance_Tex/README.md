## Texture Preparation 

When the file split name is 'new+penn+bed+07091127', run *python render_shading.py --house='penn' --floor='bed' --pano='07091127' --height=512*

The complete reflectance layer will be exported into folder 'out_obj' and named as 'color.jpg'

Copy the latest 'color.jpg' file into folder '/00_Data/zind/scenes/layout_merge' and replace the existing file.

Go to folder [03_3DFloor_Mesh](https://github.com/Gzhji/vs_natural_ill/tree/main/03_3DFloor_Mesh) and run floormesh.py --split='new+penn+bed+07091127'

The planar surfaces (from the reflectance layer) will be generated and the results will be exported into the folder '/00_Data/zind/scenes/floormesh'.
 

