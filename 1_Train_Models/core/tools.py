import os
import sys
import itertools

sys.path.append(os.path.dirname(__file__))

import config

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def test_and_create_dirs(*args):
    for fname in args:
        d, f = os.path.split(fname)
        if not os.path.exists(d):
            os.makedirs(d)

def save_dict_to_csv(save_file, d):
    test_and_create_dirs(save_file)
    with open(save_file, 'w') as fp:
        for k, v in d.items():
            kvstr = str(k) + ',' + ','.join([str(i) for i in v]) + ',\n'
            fp.write(kvstr)

def filter_by_arch_opt_levels(d):
    if not config.ARCHS[0] == 'all':
        is_arch_satisfied = False
        if '32' in config.ARCHS:
            config.ARCHS.append('86')
        for arch in config.ARCHS:
            if arch in d:
                is_arch_satisfied = True
                break
        if not is_arch_satisfied:
            return False
    if not config.OPT_LEVELS[0] == 'all':
        is_opt_level_satisfied = False
        for level in config.OPT_LEVELS:
            if level in d:
                is_opt_level_satisfied = True
                break
        if not is_opt_level_satisfied:
            return False
    return True
