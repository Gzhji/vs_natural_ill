import sys
from pathlib import Path
import lxml.etree as ET
from xml.dom import minidom
from math import isclose


class MTLParser:

    def __init__(self, fname, default_Kd=1.0):
        self.default_Kd = default_Kd
        self.props = ['Ka', 'Kd', 'Ks', 'Ke', 'Ki', 'Pm', 'Pr', 'd', 'Tr', 'Ds', 'Ni', 'Ns', 'map_Kd', 'map_Ke', 'map_Pm', 'map_Pr', 'map_d', 'map_Tr', 'norm']
        self.mdl2mtl = {'baseColor': 'Kd', 'glow': 'Ke', 'interiorColor': 'Ki', 'metallic': 'Pm', 'roughness': 'Pr', 'normal': 'norm', 'opacity': 'd', 'translucence': 'Tr', 'density': 'Ds', 'indexOfRefraction': 'Ni'}
        if fname.endswith('.mdl'):
            self._parse_mdl_file(fname)
        elif fname.endswith('.mtl'):
            self._parse_mtl_file(fname)
        elif fname.endswith('.obj'):
            f = open(fname, 'r')
            lines = [line.strip() for line in f.readlines()]
            f.close()
            parsed = False
            for line in lines:
                if line.startswith('adobe_mdllib '):
                    fname = Path(fname).parent / line.split()[1]
                    self._parse_mdl_file(fname)
                    parsed = True
                    break
            if not parsed:
                for line in lines:
                    if line.startswith('mtllib '):
                        fname = Path(fname).parent / line.split()[1]
                        self._parse_mtl_file(fname)
                        break
            if len(self.mtl_names) > 1:
                self.mtl_names = []
                for line in lines:
                    if line.startswith('usemtl '):
                        mtl_name = line.split()[1]
                        if mtl_name not in self.mtl_names:
                            self.mtl_names.append(mtl_name)
        else:
            assert(False)
        self.parent = Path(fname).parent

    def _parse_mdl_file(self, fname):
        f = open(fname, 'r')
        lines = [line.strip() for line in f.readlines()]
        f.close()
        self.mtl_names, line_ids = [], []
        for line_id, line in enumerate(lines):
            if line.startswith('export material '):
                self.mtl_names.append(line.split()[2].strip('(*)'))
                line_ids.append(line_id)
        line_ids.append(len(lines))
        self.mtls = {}
        for mtl_id, mtl_name in enumerate(self.mtl_names):
            print(mtl_name)
            self.mtls[mtl_name] = self._parse_mdl(lines[line_ids[mtl_id]:line_ids[mtl_id + 1]], mtl_id=mtl_id)

    def _parse_mdl(self, lines, mtl_id):
        if self.default_Kd is None:
            default_Kd = mtl_id / 10
        else:
            default_Kd = 1.0
        mtl = {'Ka': '0 0 0', 'Kd': f'{default_Kd} {default_Kd} {default_Kd}', 'Ks': '0 0 0'}
        for line in lines:
            for mdl_prop in self.mdl2mtl:
                if line.startswith(mdl_prop):
                    # texture
                    prop = self.mdl2mtl[mdl_prop]
                    if line.find("\"") >= 0:
                        if prop != 'norm':
                            prop = 'map_' + prop
                        mtl[prop] = line[line.find("\"")+1:line.rfind("\"")]
                    elif line.find("color(") >= 0:
                        mtl[prop] = ' '.join(line[line.find("(")+1:line.rfind(")")].split(', '))
                    elif line.find("float(") >= 0:
                        mtl[prop] = line[line.find("(")+1:line.rfind(")")]
        return mtl

    def _parse_mtl_file(self, fname):
        f = open(fname, 'r')
        lines = [line.strip(' (),\t\n') for line in f.readlines()]
        f.close()
        self.mtl_names, line_ids = [], []
        for line_id, line in enumerate(lines):
            if line.startswith('newmtl '):
                self.mtl_names.append(line.split()[1])
                line_ids.append(line_id)
        line_ids.append(len(lines))
        self.mtls = {}
        for mtl_id, mtl_name in enumerate(self.mtl_names):
            self.mtls[mtl_name] = self._parse_mtl(lines[line_ids[mtl_id]:line_ids[mtl_id + 1]])

    def _parse_mtl(self, lines):
        mtl = {'Ka': '0 0 0', 'Kd': '1 1 1', 'Ks': '0 0 0'}
        for line in lines:
            for prop in self.props:
                if line.startswith(prop):
                    mtl[prop] = ' '.join(line.split()[1:])
        return mtl

    def save_mtl(self, fname='stdout', rename=False):
        if fname == 'stdout':
            f = sys.stdout
        else:
            f = open(fname, 'w')
        for mtl_id, mtl_name in enumerate(self.mtl_names):
            mtl = self.mtls[mtl_name]
            if rename:
                print('newmtl', 'material_' + str(mtl_id), file=f)
            else:
                print('newmtl', mtl_name, file=f)
            for prop in self.props:
                if prop in mtl:
                    print(prop, mtl[prop], file=f)
        if fname != 'stdout':
            f.close()

    def save_mitsuba(self, fname='stdout'):
        xmls = []
        if len(self.mtl_names) == 1:
            mtl_name = self.mtl_names[0]
            xml = self._create_mitsuba_bsdf(self.mtls[mtl_name])
            xmls.append(xml)
        else:
            for mtl_name in self.mtl_names:
                xml = self._create_mitsuba_bsdf(self.mtls[mtl_name], mtl_name)
                xmls.append(xml)
        if fname is None:
            return xmls
        if fname == 'stdout':
            f = sys.stdout
        else:
            f = open(fname, 'w')
        print('\n'.join(xmls), file=f)
        if fname != 'stdout':
            f.close()
        return xmls

    def _new_bsdf(self, bsdf, bsdf_type):
        if bsdf is None:
            bsdf = ET.Element('bsdf')
            bsdf.set('type', bsdf_type)
            self.top_bsdf = bsdf
        else:
            bsdf = ET.SubElement(bsdf, 'bsdf')
            bsdf.set('type', bsdf_type)
        return bsdf

    def _create_mitsuba_bsdf(self, mtl, mtl_name=None):
        bsdf = None
        if 'norm' in mtl:
            bsdf = self._add_mitsuba_normalmap(bsdf, mtl)
        if 'Pr' in mtl or 'map_Pr' in mtl:
            self._add_mitsuba_rough(bsdf, mtl)
        else:
            self._add_mitsuba_diffuse(bsdf, mtl)
        if mtl_name is not None:
            self.top_bsdf.set('name', mtl_name)
        xml = minidom.parseString(ET.tostring(self.top_bsdf)).toprettyxml(indent='  ')[23:]
        return xml

    def _add_mitsuba_normalmap(self, bsdf, mtl):
        bsdf = self._new_bsdf(bsdf, 'normalmap')
        self._add_mitsuba_texture(bsdf, mtl['norm'], gamma=1.0)
        return bsdf

    def _add_mitsuba_rough(self, bsdf, mtl):
        if 'map_Tr' in mtl:
            bsdf = self._new_bsdf(bsdf, 'blendbsdf')
            self._add_mitsuba_texture(bsdf, mtl['map_Tr'], name='weight', gamma=1.0)
            self._add_mitsuba_roughreflective(bsdf, mtl)
            self._add_mitsuba_roughdielectric(bsdf, mtl)
        elif 'Tr' in mtl and not isclose(float(mtl['Tr']) , 0):
            if isclose(float(mtl['Tr']), 1):
                bsdf = self._add_mitsuba_roughdielectric(bsdf, mtl)
            else:
                bsdf = self._new_bsdf(bsdf, 'blendbsdf')
                self._add_mitsuba_float(bsdf, mtl['Tr'], name='weight')
                self._add_mitsuba_roughreflective(bsdf, mtl)
                self._add_mitsuba_roughdielectric(bsdf, mtl)
        else:
            bsdf = self._add_mitsuba_roughreflective(bsdf, mtl)
        return bsdf

    def _add_mitsuba_roughreflective(self, bsdf, mtl):
        # ignore opacity
        """
        if 'map_d' in mtl:
            bsdf = self._new_bsdf(bsdf, 'mask')
            self._add_mitsuba_texture(bsdf, mtl['map_d'], name='opacity', gamma=1.0)
            self._add_mitsuba_roughopaque(bsdf, mtl)
        elif 'd' in mtl and not isclose(float(mtl['d']) , 1):
            bsdf = self._new_bsdf(bsdf, 'mask')
            self._add_mitsuba_float(bsdf, mtl['d'], name='opacity')
            self._add_mitsuba_roughopaque(bsdf, mtl)
        else:
            bsdf = self._add_mitsuba_roughopaque(bsdf, mtl)
        """
        bsdf = self._add_mitsuba_roughopaque(bsdf, mtl)
        return bsdf

    def _add_mitsuba_roughopaque(self, bsdf, mtl, twosided=True):
        if twosided:
            bsdf = self._new_bsdf(bsdf, 'twosided')
            twosided = bsdf
        if 'map_Pm' in mtl:
            bsdf = self._new_bsdf(bsdf, 'blendbsdf')
            self._add_mitsuba_texture(bsdf, mtl['map_Pm'], name='weight', gamma=1.0)
            self._add_mitsuba_roughplastic(bsdf, mtl)
            self._add_mitsuba_roughconductor(bsdf, mtl)
        elif 'Pm' in mtl and not isclose(float(mtl['Pm']) , 0):
            if isclose(float(mtl['Pm']), 1):
                bsdf = self._add_mitsuba_roughconductor(bsdf, mtl)
            else:
                bsdf = self._new_bsdf(bsdf, 'blendbsdf')
                self._add_mitsuba_float(bsdf, mtl['Pm'], name='weight')
                self._add_mitsuba_roughplastic(bsdf, mtl)
                self._add_mitsuba_roughconductor(bsdf, mtl)
        else:
            bsdf = self._add_mitsuba_roughplastic(bsdf, mtl)
        if twosided != False:
            return twosided
        else:
            return bsdf

    def _add_mitsuba_roughdielectric(self, bsdf, mtl):
        bsdf = self._new_bsdf(bsdf, 'roughdielectric')
        if 'Ni' in mtl and float(mtl['Ni']) > 0.0 and float(mtl['Ni']) < 3.0:
            self._add_mitsuba_float(bsdf, mtl['Ni'], name='int_ior')
        if 'map_Kd' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Kd'], name='specularReflectance')
        else:
            basecolor = ET.SubElement(bsdf, 'rgb') #kd srgb
            basecolor.set('name', 'specularReflectance')
            basecolor.set('value', mtl['Kd'].replace(' ', ', '))
        if 'map_Pr' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Pr'], name='alpha', gamma=1.0)
        else:
            self._add_mitsuba_float(bsdf, mtl['Pr'], name='alpha')
        return bsdf

    def _add_mitsuba_roughplastic(self, bsdf, mtl):
        bsdf = self._new_bsdf(bsdf, 'roughplastic')
        if 'Ni' in mtl and float(mtl['Ni']) > 0.0 and float(mtl['Ni']) < 3.0:
            self._add_mitsuba_float(bsdf, mtl['Ni'], name='int_ior')
        if 'map_Kd' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Kd'], name='diffuse_reflectance')
        else:
            basecolor = ET.SubElement(bsdf, 'rgb') #kd srgb
            basecolor.set('name', 'diffuse_reflectance')
            basecolor.set('value', mtl['Kd'].replace(' ', ', '))
        if 'map_Pr' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Pr'], name='alpha', gamma=1.0)
        else:
            self._add_mitsuba_float(bsdf, mtl['Pr'], name='alpha')
        return bsdf

    def _add_mitsuba_roughconductor(self, bsdf, mtl):
        bsdf = self._new_bsdf(bsdf, 'roughconductor')
        material = ET.SubElement(bsdf, 'string')
        material.set('name', 'material')
        material.set('value', 'none')
        if 'map_Kd' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Kd'], name='specular_reflectance')#specularReflectance
        else:
            basecolor = ET.SubElement(bsdf, 'rgb') #kd srgb
            basecolor.set('name', 'specular_reflectance')#specularReflectance
            basecolor.set('value', mtl['Kd'].replace(' ', ', '))
        if 'map_Pr' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Pr'], name='alpha', gamma=1.0)
        else:
            self._add_mitsuba_float(bsdf, mtl['Pr'], name='alpha')
        return bsdf

    def _add_mitsuba_diffuse(self, bsdf, mtl, twosided=True):
        if twosided:
            bsdf = self._new_bsdf(bsdf, 'twosided')
            twosided = bsdf
        bsdf = self._new_bsdf(bsdf, 'diffuse')
        if 'map_Kd' in mtl:
            self._add_mitsuba_texture(bsdf, mtl['map_Kd'], name='reflectance')
        else:
            basecolor = ET.SubElement(bsdf, 'rgb') #srgb
            basecolor.set('name', 'reflectance')
            basecolor.set('value', mtl['Kd'].replace(' ', ', '))
        if twosided != False:
            return twosided
        else:
            return bsdf

    def _add_mitsuba_texture(self, bsdf, fname, name=None, gamma=None):
        texture = ET.SubElement(bsdf, 'texture')
        texture.set('type', 'bitmap')
        if name is not None:
            texture.set('name', name)
        tex_fname = ET.SubElement(texture, 'string')
        tex_fname.set('name', 'filename')
        tex_fname.set('value', str(self.parent / fname))
        if gamma is not None:
            tex_gamma = ET.SubElement(texture, 'float')
            tex_gamma.set('name', 'gamma')
            tex_gamma.set('value', str(gamma))
        return texture

    def _add_mitsuba_float(self, bsdf, value, name):
        quant = ET.SubElement(bsdf, 'float')
        quant.set('name', name)
        quant.set('value', value)
        return quant

