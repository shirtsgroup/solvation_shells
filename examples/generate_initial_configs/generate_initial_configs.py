# Script to slowly pull along the CN to generate initial configurations near each umbrella simulation

import numpy as np
import subprocess
from textwrap import dedent

def run(commands):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run grompp with mdp file on gro file with topology top
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if tpr is None:
        tpr = gro.split('.gro')[0] + '.tpr'
    cmd = [f'{gmx} grompp -f {mdp} -p {top} -c {gro} -o {tpr}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return tpr


def mdrun(tpr, output=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run GROMACS with tpr file
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}
    '''
    if output is None:
        output = tpr.split('.tpr')[0]
    cmd = [f'{gmx} mdrun -s {tpr} -deffnm {output}']
    
    for f in flags:
        cmd[0] += f' -{f} {flags[f]}'

    if dry_run:
        print(cmd)
    else:
        run(cmd)
    return output + '.gro'


def write_plumed(options, filename='plumed.dat'):
    '''Write plumed input file for umbrella sampling simulation'''

    f = dedent(f'''\
    water_group: GROUP ATOMS=1-3000:3   # oxygen atom of the water molecules
    n: COORDINATION GROUPA=3001 GROUPB=water_group SWITCH={{Q REF={options['R_0']} BETA=-21.497624558253246 LAMBDA=1 R_0={options['R_0']}}}
    t: MATHEVAL ARG=n FUNC=1000-x PERIODIC=NO

    r: RESTRAINT ARG=t KAPPA={options['KAPPA']} AT={options['AT']} # apply a harmonic restraint at CN=AT with force constant = KAPPA kJ/mol

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def run_plumed_driver(plumed_input, traj, dt=0.002, stride=250, output='COLVAR'):
    '''Run plumed driver on plumed input file for a given trajectory (as an xtc) and read output COLVAR'''
    if traj.split('.')[-1] == 'xtc':
        cmd = f'mpirun -np 1 plumed driver --plumed {plumed_input} --ixtc {traj} --timestep {dt} --trajectory-stride {stride}'
    elif traj.split('.')[-1] == 'gro':
        cmd = f'mpirun -np 1 plumed driver --plumed {plumed_input} --igro {traj}'
    run(cmd)

    COLVAR = np.loadtxt(output, comments='#')
    return COLVAR


if __name__ == '__main__':

    # Gromacs inputs
    equil_gro = '../prod.gro'
    top = 'solution.top'
    mdp = 'increment.mdp'
    ntasks = 16

    # PLUMED inputs
    n_sims = 16
    CN_range = (1,9)

    plumed_options = {
        'R_0' : 0.315,    # cutoff (nm)
        'AT' : 6,         # umbrella center (CN)
        'KAPPA' : 100,    # force constant
        'STRIDE' : 10,    # how often to output
        'FILE' : 'COLVAR' # filename of output
    }

    # calculate the coordination numbers to add bias at
    AT_values = np.linspace(CN_range[0], CN_range[1], n_sims)

    # get the initial state
    plumed_options['STRIDE'] = 1
    plumed_input = write_plumed(plumed_options)
    cn_equil = run_plumed_driver(plumed_input, '../prod.gro')
    print('CN at the end of equilibrium =', cn_equil[2])

    # find which values are above and below the initial state
    lt = np.where(AT_values <= cn_equil[2])[0]
    gt = np.where(AT_values >= cn_equil[2])[0]

    # increment through the values below the initial state
    gro = equil_gro
    for i in lt[::-1]:
        plumed_options['AT'] = AT_values[i]
        plumed_options['FILE'] = f'COLVAR_{i}'
        plumed_input = write_plumed(plumed_options, filename=f'plumed_{i}.dat')
        tpr = grompp(gro, mdp, top, tpr=f'prod_{i}.tpr', gmx='mpirun -np 1 gmx_mpi')
        gro = mdrun(tpr, output=f'prod_{i}', flags={'plumed' : f'plumed_{i}.dat'}, gmx=f'mpirun -np {ntasks} gmx_mpi')


     # increment through the values above the initial state
    gro = equil_gro
    for i in gt:
        plumed_options['AT'] = AT_values[i]
        plumed_options['FILE'] = f'COLVAR_{i}'
        plumed_input = write_plumed(plumed_options, filename=f'plumed_{i}.dat')
        tpr = grompp(gro, mdp, top, tpr=f'prod_{i}.tpr', gmx='mpirun -np 1 gmx_mpi')
        gro = mdrun(tpr, output=f'prod_{i}', flags={'plumed' : f'plumed_{i}.dat'}, gmx=f'mpirun -np {ntasks} gmx_mpi')