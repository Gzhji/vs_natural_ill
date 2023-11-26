#!/usr/bin/env python

__author__ = "Christopher Hahne"
__email__ = "info@christopherhahne.de"
__license__ = """
    Copyright (c) 2020 Christopher Hahne <info@christopherhahne.de>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

from color_matcher import __version__
from color_matcher.top_level import ColorMatcher, METHODS
from color_matcher.io_handler import *

import getopt
import sys, os


def usage():

    print("Usage: color-matcher <options>\n")
    print("Options:")
    print("-s <path>,     --src=<path>       Specify source image file or folder to process")
    print("-r <filepath>, --ref=<filepath>   Specify target image file")
    print("-m <method>,   --method=<method>  Provide color transfer method such as:")
    print("                                  "+', '.join(['"'+m+'"' for m in METHODS]))
    print("-w ,           --win              Select files from window")
    print("-h,            --help             Print this help message")
    print("")


def parse_options(argv):

    try:
        opts, args = getopt.getopt(argv, "hs:r:m:w", ["help", "src=", "ref=", "method=", "win"])
    except getopt.GetoptError as e:
        print(e)
        sys.exit(2)

    # create dictionary containing all parameters
    cfg = dict()

    # default settings (use test data images for MKL conversion)
    cfg['src_path'] = '.'
    cfg['ref_path'] = '.'
    cfg['method'] = METHODS[0]
    cfg['win'] = None

    if opts:
        for (opt, arg) in opts:
            if opt in ("-h", "--help"):
                usage()
                sys.exit()
            if opt in ("-s", "--src"):
                cfg['src_path'] = arg.strip(" \"\'")
            if opt in ("-r", "--ref"):
                cfg['ref_path'] = arg.strip(" \"\'")
            if opt in ("-m", "--method"):
                cfg['method'] = arg.strip(" \"\'")
            if opt in ("-w", "--win"):
                cfg['win'] = True

    return cfg


def main():

    # program info
    print("\ncolor-matcher v%s \n" % __version__)

    # retrieve parse options
    cfg = parse_options(sys.argv[1:])

    # select files from window (if option set)
    if cfg['win']:
        cfg['src_path'] = select_file(cfg['src_path'], 'Select source image')
        cfg['ref_path'] = select_file(cfg['src_path'], 'Select reference image')

    # cancel if file paths not provided
    if not cfg['src_path'] or not cfg['ref_path']:
        usage()
        print('Canceled due to missing image file path\n')
        sys.exit()

    # select image(s) considering provided folder or file
    if os.path.isdir(cfg['src_path']) and os.path.isfile(cfg['ref_path']):
        # case where source is directory and reference is file
        filenames = [os.path.join(cfg['src_path'], f) for f in os.listdir(cfg['src_path'])
                     if f.lower().endswith(FILE_EXTS)]
        output_path = os.path.join(cfg['src_path'], 'batch_proc_'+str(cfg['method']))
        os.makedirs(output_path, exist_ok=True)
        print('Output files are placed in created directory %s' % os.path.join('', os.path.basename(output_path)))
    elif os.path.isfile(cfg['src_path']) and os.path.isfile(cfg['ref_path']):
        # case where source is file and reference is file
        filenames = [cfg['src_path']]
        output_path = os.path.dirname(cfg['src_path'])
        filename = os.path.splitext(os.path.basename(cfg['src_path']))[0]+'_'+cfg['method']
        file_ext = os.path.splitext(cfg['src_path'])[-1]
        print('Output file is named %s' % os.path.join('', filename + file_ext))
    else:
        # unsupported cases
        print('File(s) not found \n')
        sys.exit()

    # method handling
    cfg['method'] = cfg['method'] if cfg['method'] in METHODS else METHODS[0]

    # read reference image
    ref = load_img_file(cfg['ref_path'])

    # process images
    for f in filenames:
        src = load_img_file(f)
        res = ColorMatcher(src=src, ref=ref, method=cfg['method']).main()
        filename = os.path.splitext(os.path.basename(f))[0]+'_'+cfg['method']
        file_ext = os.path.splitext(f)[-1]
        save_img_file(res, file_path=os.path.join(output_path, filename), file_type=file_ext[1:])

    return True


if __name__ == "__main__":

    sys.exit(main())
