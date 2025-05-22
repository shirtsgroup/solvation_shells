# functions to run Gromacs commands

import subprocess

def run(commands: list[str] | str):
    '''Run commands with subprocess'''
    if not isinstance(commands, list):
        commands = [commands]
    
    for cmd in commands:
        subprocess.run(cmd, shell=True)


def grompp(gro, mdp, top, tpr=None, gmx='gmx', flags={}, dry_run=False):
    '''
    Run grompp with mdp file on gro file with topology top and optionally, tpr

    gmx is the gmx executable as a string
    
    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}

    dry_run is a Boolean controlling whether the commands are actually run or printed to standard out
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
    Run GROMACS with tpr file using -deffnm output

    gmx is the gmx executable as a string

    flags should be a dictionary containing any additional flags, e.g. flags = {'maxwarn' : 1}

    dry_run is a Boolean controlling whether the commands are actually run or printed to standard out
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


def run_plumed(plumed_input, traj, dt=0.002, stride=250, output='COLVAR'):
    '''
    Run plumed driver on plumed input file for a given trajectory (as an xtc) and read output COLVAR'''
    cmd = f'plumed driver --plumed {plumed_input} --ixtc {traj} --timestep {dt} --trajectory-stride {stride}'
    run(cmd)

    COLVAR = np.loadtxt(output, comments='#')
    return COLVAR