# Script to run GROMACS with PLUMED umbrella sampling simulations

import argparse

from utils.gromacs_utils import run, grompp, mdrun

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--sim_num', type=int, help='Number of the umbrella sampling simulation')
    parser.add_argument('-g', '--gro', type=str, default='frame40001.gro', help='Input coordinate file for umbrella simulation')
    parser.add_argument('-m', '--mdp', type=str, default='prod.mdp', help='Input mdp file for umbrella simulation')
    parser.add_argument('-p', '--top', type=str, default='solution.top', help='Topology file for umbrella simulation')
    parser.add_argument('-n', '--ntasks', type=int, default=16, help='Number of processors to use with mpirun')
    args = parser.parse_args()

    # run Gromacs for each umbrella simulation
    tpr = grompp(args.gro, args.mdp, args.top, tpr=f'prod_{args.sim_num}.tpr', gmx='mpirun -np 1 gmx_mpi')
    mdrun(tpr, output=f'prod_{args.sim_num}', flags={'plumed' : f'plumed_{args.sim_num}.dat'}, gmx=f'mpirun -np {args.ntasks} gmx_mpi')


