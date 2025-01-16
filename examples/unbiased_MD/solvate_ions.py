# Script to run simulations of solvated ions

import argparse
from textwrap import dedent

from utils.gromacs_utils import run, grompp, mdrun
from utils.file_rw import write_packmol


def update_topology(n_waters, n_cations, n_anions, water_path='../../top/forOPC3/OPC3.itp', ions_path='../../top/forOPC3/monovalent/ions.itp', filename='solution.top'):
    '''Update the top file with new numbers of waters and ions'''

    f = dedent(f'''\
    [ defaults ]
    ; nbfunc        comb-rule       gen-pairs       fudgeLJ fudgeQQ
    1               2               yes             0.5     0.8333

    [ atomtypes ]
    ;type, bondingtype, atomic_number, mass, charge, ptype, sigma, epsilon
    OW   OW     8     0.00000000  0.00000000  A     3.17427e-01  6.8369e-01
    HW   HW     1     0.00000000  0.00000000  A     0.00000e+00  0.00000e+00
    Li   Li  	 3	   6.94000000	0.00000000	A	2.26288274e-01	1.36251960e-02
    Na   Na  	 11    22.99000000	0.00000000	A	2.61746043e-01	1.26028774e-01
    K    K   	 19    39.10000000	0.00000000	A	3.03440103e-01	5.86672238e-01
    Rb   Rb  	 37    85.47000000	0.00000000	A	3.20723539e-01	8.91730690e-01
    Cs   Cs  	 55    132.91000000	0.00000000	A	3.50123196e-01	1.49632371e+00
    Tl   Tl  	 81    204.38000000	0.00000000	A	2.99698329e-01	5.28388699e-01
    Cu   Cu  	 29    63.55000000	0.00000000	A	2.13993872e-01	4.69863200e-03
    Ag   Ag  	 47    107.87000000	0.00000000	A	2.37869958e-01	3.18714108e-02
    F    F   	 9	    9.00000000	0.00000000	A	3.23930774e-01	9.53762046e-01
    Cl   Cl  	 17    35.45000000	0.00000000	A	4.10882489e-01	2.68724396e+00
    Br   Br  	 35    79.90000000	0.00000000	A	4.45627539e-01	3.18294026e+00
    I    I   	 53    126.90000000	0.00000000	A	4.95339687e-01	3.64743606e+00 
    
    #include "{water_path}"
    #include "{ions_path}"

    [ system ]
    water and NaCl

    [ molecules ]
    ; Compound        nmols
    SOL              {n_waters}
    NA               {n_cations}
    CL				 {n_anions}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


if __name__ == '__main__':

    # parse inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conc', type=float, default=0.2, help='Concentration to which to solvate')
    parser.add_argument('-g', '--gro', type=str, default='solution.gro', help='Name of the coordinate file to generate')
    parser.add_argument('-mm', '--min', type=str, default='min.mdp', help='Input mdp file for minimization')
    parser.add_argument('-mv', '--nvt', type=str, default='nvt.mdp', help='Input mdp file for NVT equilibration')
    parser.add_argument('-mp', '--npt', type=str, default='npt.mdp', help='Input mdp file for NPT equilibration')
    parser.add_argument('-md', '--prod', type=str, default='prod.mdp', help='Input mdp file for production')
    parser.add_argument('-p', '--top', type=str, default='solution.top', help='Topology file')
    parser.add_argument('-i', '--inp', type=str, default='solution.inp', help='Packmol input file for creating initial box')
    parser.add_argument('-n', '--ntasks', type=int, default=16, help='Number of processors to use with mpirun')
    args = parser.parse_args()    

    # run packmol to place ions in water
    inp = write_packmol(args.conc, '../../inputs/unbiased_MD/Na.pdb', '../../inputs/unbiased_MD/Cl.pdb', cation_charge=1, anion_charge=-1,
                        n_waters=1107, water='../../inputs/unbiased_MD/water.pdb', filename=args.inp, top=args.top)
    cmd = f'packmol < {inp}'
    run(cmd)

    # run gmx editconf to convert pdb to gro
    cmd = f'mpirun -np 1 gmx_mpi editconf -f solution.pdb -o {args.gro} -bt cubic -box 3.2 3.2 3.2'
    run(cmd)

    # minimization
    min_tpr = grompp(args.gro, args.min, top=args.top, tpr='min.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    min_gro = mdrun(min_tpr, output='min', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NVT equilibration (50 ps)
    nvt_tpr = grompp(min_gro, args.nvt, top=args.top, tpr='nvt.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    nvt_gro = mdrun(nvt_tpr, output='nvt', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NPT equilibration (1 ns)
    npt_tpr = grompp(nvt_gro, args.npt, top=args.top, tpr='npt.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    npt_gro = mdrun(npt_tpr, output='npt', gmx=f'mpirun -np {args.ntasks} gmx_mpi')

    # NPT production (20 ns)
    prod_tpr = grompp(npt_gro, args.prod, top=args.top, tpr='prod.tpr', gmx=f'mpirun -np 1 gmx_mpi')
    prod_gro = mdrun(prod_tpr, output='prod', gmx=f'mpirun -np {args.ntasks} gmx_mpi')