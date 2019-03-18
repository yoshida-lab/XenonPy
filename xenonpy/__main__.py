#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import argparse
from os import remove, rename
from pathlib import Path

from tqdm import tqdm

from .utils import config


def migrate(args_):
    data_path = Path('~/.xenonpy/dataset').expanduser().absolute()
    user_path = Path(config('userdata')).expanduser().absolute()

    def migrate_(f):
        path = data_path / f
        if not path.exists():
            return
        if args_.keep:
            path_ = user_path / f
            rename(str(path), str(path_))
        else:
            remove(str(path))

    for file in tqdm(['mp_inorganic.pkl.pd_', 'mp_structure.pkl.pd_', 'oqmd_inorganic.pkl.pd_',
                      'oqmd_structure.pkl.pd_'], desc='migrating'):
        migrate_(file)


parser = argparse.ArgumentParser(
    prog='XenonPy',
    description='''
    XenonPy is a Python library that implements a comprehensive set of
    machine learning tools for materials informatics.
    ''')
subparsers = parser.add_subparsers()

parser_migrate = subparsers.add_parser('migrate', help='see `migrate -h`')
parser_migrate.add_argument(
    '-k',
    '--keep',
    action='store_true',
    help=
    'Keep files fetched from `yoshida-lab/dataset`. These files will be moved to `userdata` dir.'
)
parser_migrate.set_defaults(handler=migrate)

args = parser.parse_args()
if hasattr(args, 'handler'):
    args.handler(args)
else:
    parser.print_help()
