# Script to run on Bridges-2 to submit parallel umbrella simulations

import numpy as np
import MDAnalysis as mda

from utils.gromacs_utils import run
from utils.file_rw import write_plumed_decoordination, write_sbatch_umbrella

if __name__ == '__main__':

    # create MDAnalysis groups for plumed and write to Gromacs ndx file
    u = mda.Universe('mda_readable.tpr', 'frame40001.gro')
    ion = u.select_atoms('resname NA')[0]
    not_ion = u.select_atoms('not element H') - ion

    mda.AtomGroup([ion]).write('index.ndx', mode='w', name='ion')
    not_ion.write('index.ndx', mode='a', name='not_ion')

    # PLUMED inputs
    n_sims = 16
    AT_values = np.linspace(1,9, n_sims)

    plumed_options = {
        'ndx'           : 'index.ndx',
        'ion_group'     : 'ion',
        'not_ion_group' : 'not_ion',
        'R_0'           : 0.315,
        'a'             : 21.497624558253246,
        'AT'            : 6,
        'KAPPA'         : 100,
        'n_group'       : len(not_ion),
        'STRIDE'        : 10,
        'FILE'          : 'COLVAR' 
    }

    # SBATCH inputs
    sbatch_options = {
        'sim_num' : 0,
        'ntasks' : 16,
        'time' : '16:00:00',
        'job' : 'umbrella_Cdilute_all-molecule',
        'gro' : 'frame40001.gro',
        'top' : 'solution.top',
        'mdp' : 'prod.mdp'
    }

    # write input files and submit
    for i in range(n_sims):
        plumed_options['AT'] = AT_values[i]
        plumed_options['FILE'] = f'COLVAR_{i}'
        sbatch_options['sim_num'] = i

        plumed = write_plumed_decoordination(plumed_options, filename=f'plumed_{i}.dat')
        sbatch = write_sbatch_umbrella(sbatch_options, filename=f'submit_umbrella_{i}.job')

        cmd = f'sbatch {sbatch}'
        run(cmd)