import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

import argparse
import sh
import lxml.etree as ET
from xml.dom import minidom
import cv2
from utils.misc import *
import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='sample')
    parser.add_argument('--log', type=str, default='out')
    parser.add_argument('--outdir', type=str, default='../data/zind/scenes/arch')
    parser.add_argument('--height', type=int, default=1024)

    opt = parser.parse_args()
    remkdir(opt.log)
    return opt


def parse_xml(xml, **kwargs):
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


def render_mitsuba(scene, xml_fname='scene', render_fname='render'):
    xml = write_xml(scene)
    print_log(xml, fname=f'{opt.log}/{xml_fname}.xml', screen=False, mode='w')
    print_log(f'mitsuba {opt.log}/{xml_fname}.xml -o {opt.log}/{render_fname}', fname=f'{opt.log}/commands.sh', screen=False, mode='w')
    sh.bash(f'{opt.log}/commands.sh')


def build_arch(split_id, house_id, floor_id, pano_id, mesh_id):
    mesh = ET.Element('shape')
    mesh.set('type', 'obj')
    mesh.append(parse_xml(f'<string name="filename" value="../data/zind/scenes/floormesh/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}/{mesh_id}.obj"/>'))
    transform = parse_xml(f'configs/transform/toworld.xml')
    mesh.append(transform)
    return mesh


def render_arch(opt):

    for line in tqdm.tqdm(open(f'../../lists/zind_panos_{opt.split}.txt').readlines()):
        split_id, house_id, floor_id, pano_id = line.strip().split()
        if os.path.exists(f'{opt.outdir}/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}_arch.png'):
            print(split_id, house_id, floor_id, pano_id)
            continue
        scene = parse_xml('configs/scenes/scene.xml')
        scene.append(parse_xml('configs/sensors/hdr_box.xml', height=opt.height, width=opt.height * 2, samples=1))
        scene.append(parse_xml('configs/integrators/shapeindex.xml', undefined=2))

        scene.append(build_arch(split_id, house_id, floor_id, pano_id, 'geo_wall'))
        scene.append(build_arch(split_id, house_id, floor_id, pano_id, 'geo_floor'))
        scene.append(build_arch(split_id, house_id, floor_id, pano_id, 'geo_ceiling'))
        render_mitsuba(scene, render_fname='arch')
        render = cv2.imread(f'{opt.log}/arch.exr', -1)
        os.makedirs(f'{opt.outdir}/{split_id}', exist_ok=True)

        cv2.imwrite(f'{opt.outdir}/{split_id}/{split_id}+{house_id}+{floor_id}+{pano_id}_arch.png', render[:,:,0] * 128)

def render_pos(opt):
    scene = parse_xml('configs/scenes/scene.xml')
    scene.append(parse_xml('configs/sensors/hdr_box.xml', height=opt.height, width=opt.height * 2, samples=1))
    scene.append(parse_xml('configs/integrators/position.xml', undefined=10000))
    scene.append(build_arch(opt.split, opt.house, opt.floor, opt.pano, 'geo_wall'))
    scene.append(build_arch(opt.split, opt.house, opt.floor, opt.pano, 'geo_floor'))
    scene.append(build_arch(opt.split, opt.house, opt.floor, opt.pano, 'geo_ceiling'))
    render_mitsuba(scene, render_fname='pos')
    render = cv2.imread(f'{opt.log}/pos.exr', -1)
    cv2.imwrite(f'{opt.log}/pos.png', (render + 0.5) * 255)

if __name__ == '__main__':
    opt = parse_args()
    print(opt)
    render_arch(opt)
    # render_pos(opt)

