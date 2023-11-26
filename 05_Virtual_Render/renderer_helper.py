import sys
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'
import cv2
import kornia.morphology
import torch
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')
sys.path.append('..')
import numpy as np
import argparse
import sh
from scipy import ndimage as nd
from utils_app.transform import *
from utils_app.mtlparser import MTLParser
sys.path.append('../03_3DFloor_Mesh')
from floormesh import *
import lxml.etree as ET
from xml.dom import minidom
from guided_filter_pytorch.guided_filter import GuidedFilter


def parse_xml(xml, **kwargs):
    '''
    parse xml
    '''
    if xml.endswith('.xml'):
        scene = ET.parse(xml, ET.XMLParser(remove_blank_text=True)).getroot()
    else:
        scene = ET.fromstring(xml, ET.XMLParser(remove_blank_text=True))
    for k1, v1 in kwargs.items():
        for elem in scene.iter():
            for k2, v2 in elem.attrib.items():
                if v2 == '$' + k1:
                    elem.attrib[k2] = str(v1)
    return scene

def write_xml(scene):
    return minidom.parseString(ET.tostring(scene)).toprettyxml(indent='    ')[23:]

def build_energy_conservation():
    return ET.fromstring('<boolean name="ensureEnergyConservation" value="false"/>')

def build_arch(opt, house_id, floor_id, pano_id, mesh_id, bsdf_id, position=None, texture=None, roughness=0.03, surface_tex = None,
               mesh_dir=None, uvscale=None, ground_floor=False):
    mesh = ET.Element('shape')
    mesh.set('type', 'obj')
    if mesh_dir is None:
        mesh.append(parse_xml(f'<string name="filename" value="../00_Data/zind/scenes/floormesh/{house_id}+{floor_id}+{pano_id}/{mesh_id}.obj"/>'))
    else:
        mesh.append(parse_xml(f'<string name="filename" value="{mesh_dir}.obj"/>'))

    to_world_path = f'../data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/toworld_arch.xml'
    if os.path.exists(to_world_path):
        transform = parse_xml(to_world_path)
    else:
        if not ground_floor:
            transform = parse_xml(f'configs/transform/toworld.xml')
        else:
            transform = parse_xml(f'configs/transform/toworld_ucsd.xml')


    if position is not None:
        transform.append(parse_xml(f'<translate x="{position[0]}" y="{position[1]}" z="{position[2]}" />'))
    mesh.append(transform)
    #
    if bsdf_id == 'white_ceiling':
        bsdf = parse_xml('configs/bsdfs/white_ceiling.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'white_floor':
        bsdf = parse_xml('configs/bsdfs/white_floor.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'white_wall':
        bsdf = parse_xml('configs/bsdfs/white_wall.xml')
    elif bsdf_id == 'white':
        bsdf = parse_xml('configs/bsdfs/white.xml', tex=surface_tex)
    if bsdf_id == 'White_Painted_Wall':
        bsdf = parse_xml('configs/bsdfs/White_Painted_Wall.xml', tex=surface_tex)
    if bsdf_id == 'White_Painted_Ceil':
        bsdf = parse_xml('configs/bsdfs/White_Painted_Ceil.xml', tex=surface_tex)
    elif bsdf_id == 'black':
        bsdf = parse_xml('configs/bsdfs/black.xml')
    elif bsdf_id == 'Carpet_E14_526':
        bsdf = parse_xml('configs/bsdfs/Carpet_E14_526.xml')
    elif bsdf_id == 'Wooden_Parquet_Floor':
        bsdf = parse_xml('configs/bsdfs/Wooden_Parquet_Floor.xml', tex=surface_tex)

    elif bsdf_id == 'trans':
        bsdf = parse_xml('configs/bsdfs/trans.xml')
    elif bsdf_id == 'mirror':
        bsdf = parse_xml('configs/bsdfs/mirror.xml')
    elif bsdf_id == 'specular':
        bsdf = parse_xml('configs/bsdfs/roughmirror.xml', roughness=roughness)
    elif bsdf_id == 'mask':
        bsdf = parse_xml('configs/bsdfs/mask.xml', texture=texture, tex=surface_tex)
    elif bsdf_id == 'mask_obj':
        bsdf = parse_xml('configs/bsdfs/mask_obj.xml', texture=texture, tex=surface_tex)
    elif bsdf_id == 'mask_r':
        bsdf = parse_xml('configs/bsdfs/mask_r.xml', texture=texture, tex=surface_tex)
    elif bsdf_id == 'mask_base':
        bsdf = parse_xml('configs/bsdfs/mask_base.xml', texture=texture)
    elif bsdf_id == 'difftrans':
        bsdf = parse_xml('configs/bsdfs/difftrans.xml', texture=texture)
    elif bsdf_id == 'area':
        bsdf = parse_xml('configs/emitters/area.xml')
    elif bsdf_id == 'envmap_tex':
        bsdf = parse_xml('configs/emitters/envmap_tex.xml', texture=texture)
    elif bsdf_id == 'occlusion_mask':
        bsdf = parse_xml('configs/bsdfs/occlusionmask.xml')
    elif bsdf_id == 'plastic':
        bsdf = parse_xml('configs/bsdfs/roughplastic.xml', roughness=roughness, texture=texture)
    elif bsdf_id == 'diffuse':
        bsdf = parse_xml('configs/bsdfs/diffuse.xml', texture=texture)
    elif bsdf_id == 'diffuse_uvscale':
        bsdf = parse_xml('configs/bsdfs/diffuse_uvscale.xml', texture=texture, uvscale=uvscale)
    if bsdf_id is not None:
        mesh.append(bsdf)
    return mesh



#
def build_sphere(opt, house_id, floor_id, pano_id, mesh_id, mesh_dir=None, position=None, texture=None, ground_floor=False):
    mesh = ET.Element('shape')
    mesh.set('type', 'obj')
    if mesh_dir is None:
        mesh.append(parse_xml(f'<string name="filename" value="../00_Data/zind/scenes/floormesh/{mesh_id}.obj"/>'))
    else:
        mesh.append(parse_xml(f'<string name="filename" value="hem_orig.obj"/>'))

    to_world_path = f'../00_Data/zind/scenes/floormesh/{opt.house}+{opt.floor}+{opt.pano}/toworld_arch.xml'
    if os.path.exists(to_world_path):
        transform = parse_xml(to_world_path)
    else:
        if not ground_floor:
            transform = parse_xml(f'configs/transform/toworld_hemi.xml')
        else:
            transform = parse_xml(f'configs/transform/toworld.xml')

    if position is not None:
        transform.append(parse_xml(f'<translate x="{position[0]}" y="{position[1]}" z="{position[2]}" />'))
    mesh.append(transform)

    env = parse_xml('configs/emitters/area.xml', texture=texture)
    mesh.append(env)

    return mesh

def build_furn(house_id, floor_id, pano_id, mesh_id, bsdf_id, position=None,
               texture=None, ground_floor=False):

    mesh = ET.Element('shape')
    mesh.set('type', 'obj')
    mesh.append(parse_xml(f'<string name="filename" value="{mesh_id}.obj"/>'))

    to_world_path = f'../data/zind/scenes/floormesh/{house_id}+{floor_id}+{pano_id}/toworld.xml'
    if os.path.exists(to_world_path):
        transform = parse_xml(to_world_path)
    else:
        if not ground_floor:
            transform = parse_xml(f'configs/transform/toworld.xml')
        else:
            transform =  parse_xml(f'configs/transform/toworld_ucsd.xml')

    if position is not None:
        transform.append(parse_xml(f'<translate x="{position[0]}" y="{position[1]}" z="{position[2]}" />'))
    mesh.append(transform)


    if bsdf_id == 'white_ceiling':
        bsdf = parse_xml('configs/bsdfs/white_ceiling.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'white_floor':
        bsdf = parse_xml('configs/bsdfs/white_floor.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'white_wall':
        bsdf = parse_xml('configs/bsdfs/white_wall.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'black':
        bsdf = parse_xml('configs/bsdfs/black.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'white':
        bsdf = parse_xml('configs/bsdfs/white.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'trans':
        bsdf = parse_xml('configs/bsdfs/trans.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'mirror':
        bsdf = parse_xml('configs/bsdfs/mirror.xml')
        mesh.append(bsdf)
    elif bsdf_id == 'full':
        xmls = MTLParser(f'{mesh_id}.mtl').save_mitsuba(None)
        for xml in xmls:
            bsdf = parse_xml(xml)
            print('bsdf', bsdf)
            mesh.append(bsdf)

    return mesh


def write_mitsuba(opt, scene, xml_fname='scene'):
    xml = write_xml(scene)
    print_log(xml, fname=f'{opt.log}/{xml_fname}.xml', screen=False, mode='w')
    return xml


def add_furns(opt, scene, bsdf_id = 'full', ground_floor=False):
    data_path = f'../00_Data/manualfill/{opt.house}+{opt.floor}+{opt.pano}/'
    print('furn path', data_path)

    for obj in Path(data_path).glob('*.obj'):
        print('obj:', obj)
        mesh_id = obj.name[:-4]
        print('mesh_id:', mesh_id)
        scene.append(build_furn(opt.house, opt.floor, opt.pano, str(obj)[:-4], bsdf_id, ground_floor=ground_floor))


def render_texture(opt, envmap_hdr_path, bright_wall_tex, wall_mesh_dir,
                      bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, furn=True, maxd=None,
                      floor_bsdf='white', wall_bsdf='white', ceil_bsdf='white', furn_bsdf='full',
                      floor_tex = None, wall_tex = None, ceil_tex = None,
                      samples = 16, wall_extend = True, hem_pos = np.array([0,0,0])):

    scene = parse_xml('configs/scenes/scene3.0.xml')
    scene.append(parse_xml('configs/sensors/hdr_envmap.xml', height=opt.height, width=opt.height * 2, samples=samples))
    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))

    scene.append(build_arch(opt, opt.house, opt.floor, opt.pano, 'tex_floor', floor_bsdf, surface_tex= floor_tex))
    scene.append(build_arch(opt, opt.house, opt.floor, opt.pano, 'tex_ceiling', ceil_bsdf, surface_tex = ceil_tex))
    scene.append(build_arch(opt, None, None, None, None, 'mask', mesh_dir= wall_mesh_dir, surface_tex = wall_tex,
                            position=np.array([0, 0, 0]), texture=f'../03_Reflectance_Tex/{opt.log}/1_win_mask_tex_img.png'))

    if wall_extend:
        scene.append(build_arch(opt, None, None, None, None, 'mask_r',
                                mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_rt_wall_mesh', surface_tex=wall_tex,
                                position=np.array([0, 0.2, 0]), texture=f'../03_Reflectance_Tex/{opt.cache}/3_bright_wall_tex_mask.png'))

    scene.append(build_sphere(opt, opt.house, opt.floor, opt.pano, 'hem_orig', position=hem_pos,
                              texture=envmap_hdr_path))

    if furn:
        print('furn is added')
        add_furns(opt, scene, furn_bsdf)

    xml = write_mitsuba(opt, scene, xml_fname='scene_hem')
    return xml

def render_obj_mask(opt, envmap_hdr_path, bright_wall_tex, wall_mesh_dir,
                      bright_rt_wall_mesh_dir, dark_rt_wall_mesh_dir, furn=True, maxd=None,
                      floor_bsdf='white', wall_bsdf='white', ceil_bsdf='white', furn_bsdf='full',
                      floor_tex = None, wall_tex = None, ceil_tex = None,
                      samples = 16, wall_extend = True, hem_pos = np.array([0,0,0])):

    scene = parse_xml('configs/scenes/scene3.0.xml')
    scene.append(parse_xml('configs/sensors/hdr_envmap.xml', height=opt.height, width=opt.height * 2, samples=samples))
    if maxd is None:
        scene.append(parse_xml('configs/integrators/volpath.xml'))
    else:
        scene.append(parse_xml('configs/integrators/volpath_depth.xml', maxd=maxd))

    scene.append(build_arch(opt, opt.house, opt.floor, opt.pano, 'tex_floor', floor_bsdf, surface_tex= floor_tex))
    scene.append(build_arch(opt, opt.house, opt.floor, opt.pano, 'tex_ceiling', ceil_bsdf, surface_tex = ceil_tex))
    scene.append(build_arch(opt, None, None, None, None, 'mask_obj', mesh_dir= wall_mesh_dir, surface_tex = wall_tex,
                            position=np.array([0, 0, 0]), texture=f'../03_Reflectance_Tex/{opt.log}/1_win_mask_tex_img.png'))

    if wall_extend:
        scene.append(build_arch(opt, None, None, None, None, 'mask_r',
                                mesh_dir = f'../03_Reflectance_Tex/{opt.cache}/3_bright_rt_wall_mesh', surface_tex=wall_tex,
                                position=np.array([0, 0.2, 0]), texture=f'../03_Reflectance_Tex/{opt.cache}/3_bright_wall_tex_mask.png'))

    scene.append(build_sphere(opt, opt.house, opt.floor, opt.pano, 'hem_orig', position=hem_pos,
                              texture=envmap_hdr_path))

    if furn:
        print('furn is added')
        add_furns(opt, scene, furn_bsdf)

    xml = write_mitsuba(opt, scene, xml_fname='scene_obj')
    return xml