# rename prod_* to init_*

from solvation_shells.utils.gromacs_utils import run

exts = ['cpt', 'edr', 'gro', 'log', 'xtc']

for i in range(16):
    for ext in exts:
        cmd = f'mv prod_{i}.{ext} init_{i}.{ext}'
        print(cmd)