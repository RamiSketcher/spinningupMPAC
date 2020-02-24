# (Rami) Created, Feb 21, 2020

import argparse
import json
import os, subprocess, sys
import os.path as osp
import string
from copy import deepcopy # copy.deepcopy?
from textwrap import dedent

import spinup


cmd = sys.argv[1] if len(sys.argv) > 1 else 'help'
print(cmd)

algo = eval('spinup.'+cmd)
print(algo)

runfile = osp.join(osp.abspath(osp.dirname(__file__)), 'utils', cmd +'.py')
print(runfile)

args = [sys.executable if sys.executable else 'python', runfile] + sys.argv[2:]
print(args)

