#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import argparse
from os import remove, rename
from pathlib import Path

from .utils import config


def migrate(args_):
    inorganic = Path('~/.xenonpy/dataset').expanduser().resolve() / 'mp_inorganic.pkl.pd_'
    structure = Path('~/.xenonpy/dataset').expanduser().resolve() / 'mp_structure.pkl.pd_'

    if args_.keep:
        userdata = config('userdata')
        inorganic_ = Path(userdata).expanduser().resolve() / 'mp_inorganic.pkl.pd_'
        structure_ = Path(userdata).expanduser().resolve() / 'mp_structure.pkl.pd_'
        rename(str(inorganic), str(inorganic_))
        rename(str(structure), str(structure_))
    else:
        remove(str(structure))
        remove(str(inorganic))


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
    'keep files `mp_inorganic.pkl.pd_` and `mp_structure.pkl.pd_`. These will be moved to `userdata` dir.'
)
parser_migrate.set_defaults(handler=migrate)

args = parser.parse_args()
if hasattr(args, 'handler'):
    args.handler(args)
else:
    parser.print_help()
