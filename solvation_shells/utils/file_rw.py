# functions to read and write files

from textwrap import dedent
import pickle

class vdW_radii:
    def __init__(self):
        self.dict = {
            "H": 1.20,  # Hydrogen
            "He": 1.40,  # Helium
            "Li": 1.82,  # Lithium
            "Be": 1.53,  # Beryllium
            "B": 1.92,  # Boron
            "C": 1.70,  # Carbon
            "N": 1.55,  # Nitrogen
            "O": 1.52,  # Oxygen
            "F": 1.47,  # Fluorine
            "Ne": 1.54,  # Neon
            "Na": 2.27,  # Sodium
            "Mg": 1.73,  # Magnesium
            "Al": 1.84,  # Aluminum
            "Si": 2.10,  # Silicon
            "P": 1.80,  # Phosphorus
            "S": 1.80,  # Sulfur
            "Cl": 1.75,  # Chlorine
            "Ar": 1.88,  # Argon
            "K": 2.75,  # Potassium
            "Ca": 2.31,  # Calcium
            "Sc": 2.11,  # Scandium
            "Ti": 2.00,  # Titanium
            "V": 2.00,  # Vanadium
            "Cr": 2.00,  # Chromium
            "Mn": 2.00,  # Manganese
            "Fe": 2.00,  # Iron
            "Co": 2.00,  # Cobalt
            "Ni": 1.63,  # Nickel
            "Cu": 1.40,  # Copper
            "Zn": 1.39,  # Zinc
            "Ga": 1.87,  # Gallium
            "Ge": 2.11,  # Germanium
            "As": 1.85,  # Arsenic
            "Se": 1.90,  # Selenium
            "Br": 1.85,  # Bromine
            "Kr": 2.02,  # Krypton
            "Rb": 3.03,  # Rubidium
            "Sr": 2.49,  # Strontium
            "Y": 2.30,  # Yttrium
            "Zr": 2.15,  # Zirconium
            "Nb": 2.00,  # Niobium
            "Mo": 2.00,  # Molybdenum
            "Tc": 2.00,  # Technetium
            "Ru": 2.00,  # Ruthenium
            "Rh": 2.00,  # Rhodium
            "Pd": 1.63,  # Palladium
            "Ag": 1.72,  # Silver
            "Cd": 1.58,  # Cadmium
            "In": 1.93,  # Indium
            "Sn": 2.17,  # Tin
            "Sb": 2.06,  # Antimony
            "Te": 2.06,  # Tellurium
            "I": 1.98,  # Iodine
            "Xe": 2.16,  # Xenon
            "Cs": 3.43,  # Cesium
            "Ba": 2.68,  # Barium
            "La": 2.50,  # Lanthanum
            "Ce": 2.48,  # Cerium
            "Pr": 2.47,  # Praseodymium
            "Nd": 2.45,  # Neodymium
            "Pm": 2.43,  # Promethium
            "Sm": 2.42,  # Samarium
            "Eu": 2.40,  # Europium
            "Gd": 2.38,  # Gadolinium
            "Tb": 2.37,  # Terbium
            "Dy": 2.35,  # Dysprosium
            "Ho": 2.33,  # Holmium
            "Er": 2.32,  # Erbium
            "Tm": 2.30,  # Thulium
            "Yb": 2.28,  # Ytterbium
            "Lu": 2.27,  # Lutetium
            "Hf": 2.25,  # Hafnium
            "Ta": 2.20,  # Tantalum
            "W": 2.10,  # Tungsten
            "Re": 2.05,  # Rhenium
            "Os": 2.00,  # Osmium
            "Ir": 2.00,  # Iridium
            "Pt": 1.75,  # Platinum
            "Au": 1.66,  # Gold
            "Hg": 1.55,  # Mercury
            "Tl": 1.96,  # Thallium
            "Pb": 2.02,  # Lead
            "Bi": 2.07,  # Bismuth
            "Po": 1.97,  # Polonium
            "At": 2.02,  # Astatine
            "Rn": 2.20,  # Radon
            "Fr": 3.48,  # Francium
            "Ra": 2.83,  # Radium
            "Ac": 2.60,  # Actinium
            "Th": 2.40,  # Thorium
            "Pa": 2.00,  # Protactinium
            "U": 1.86,  # Uranium
            "Np": 2.00,  # Neptunium
            "Pu": 2.00,  # Plutonium
            "Am": 2.00,  # Americium
            "Cm": 2.00,  # Curium
            "Bk": 2.00,  # Berkelium
            "Cf": 2.00,  # Californium
            "Es": 2.00,  # Einsteinium
            "Fm": 2.00,  # Fermium
            "Md": 2.00,  # Mendelevium
            "No": 2.00,  # Nobelium
            "Lr": 2.00   # Lawrencium
        }

    def get_dict(self):
        return self.dict


def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def write_packmol(C, cation, anion, cation_charge=1, anion_charge=-1, n_waters=1000, water='water.pdb', filename='solution.inp', packmol_options=None):
    '''
    Write packmol input file for standard MD simulation of a salt in water
    
    Parameters
    ----------
    C : float
        Concentration of the solution
    cation : str
        Filename of the cation in solution
    anion : str
        Filename of the anion in solution
    cation_charge : int
        Charge on the cation, default=+1
    anion_charge : int
        Charge on the anion, default=-1
    n_waters : int
        Number of water molecules to pack in system, default=1000
    water : str
        Filename of the water molecule in solution, default='water.pdb'
    filename : str
        Filename for the packmol input file
    packmol_options : dict
        Additional options to put in the packmol input file, default=None uses some presets.
        If specified, should have 'seed', 'tolerance', 'filetype', 'output', 'box'.

    Returns
    -------
    filename : str
        Filename of the packmol input file

    '''

    if packmol_options is None:
        packmol_options = {
            'seed' : 123456,
            'tolerance' : 2.0,
            'filetype' : 'pdb',
            'output' : filename.split('.inp')[0],
            'box' : 32
        }

    # (n_cations / salt molecule) * (N_A salt molecules / mol salt) * (C mol salt / L water) * (L water / rho g water) * (18.02 g / mol) * (mol / N_A molecules) * (n_water molecules)
    n_cations = (-cation_charge/anion_charge) * C / 997 * 18.02 * n_waters
    n_anions = (-anion_charge/cation_charge) * C / 997 * 18.02 * n_waters

    print(f'For a box of {n_waters} waters, would need {n_cations} cations and {n_anions} anions.')

    n_cations = round(n_cations)
    n_anions = round(n_anions)
    print(f'So, adding {n_cations} cations and {n_anions} anions...')

    f = dedent(f'''\
    #
    # A mixture of water and salt
    #

    # All the atoms from diferent molecules will be separated at least {packmol_options['tolerance']}
    # Anstroms at the solution.

    seed {packmol_options['seed']}
    tolerance {packmol_options['tolerance']}
    filetype {packmol_options['filetype']}

    # The name of the output file

    output {packmol_options['output']}

    # {n_waters} water molecules and {n_cations} cations, {n_anions} anions will be put in a box
    # defined by the minimum coordinates x, y and z = 0. 0. 0. and maximum
    # coordinates {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. That is, they will be put in a cube of side
    # {packmol_options['box']} Angstroms. (the keyword "inside cube 0. 0. 0. {packmol_options['box']}.") could be used as well.

    structure {water}
    number {n_waters}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}. 
    end structure

    structure {cation}
    number {n_cations}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure

    structure {anion}
    number {n_anions}
    inside box 0. 0. 0. {packmol_options['box']}. {packmol_options['box']}. {packmol_options['box']}.
    end structure


    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_plumed_dehydration(options, filename='plumed.dat'):
    '''
    Write plumed input file for umbrella sampling simulation biasing in the water-only coordination number

    Parameters
    ----------
    options : dict
        Options for the Plumed inputs. Required keys are 'N_WATERS', 'R_0', 'KAPPA', 'AT', 'STRIDE', and 'FILE'.
        'N_WATERS' is the number of water molecules in the system.
        'R_0' is the hydration shell cutoff in nm with more detail at https://www.plumed.org/doc-v2.9/user-doc/html/switchingfunction.html
        'KAPPA' and 'AT' are the force constant and the umbrella center with more detail at https://www.plumed.org/doc-v2.9/user-doc/html/_r_e_s_t_r_a_i_n_t.html
        'STRIDE' and 'FILE' are the output options for the COLVAR file, frequency of output and name of the output file.
    filename : str
        Name of the Plumed input file, default='plumed.dat'
    
    '''

    f = dedent(f'''\
    water_group: GROUP ATOMS=1-{options['N_WATERS']*3}:3   # oxygen atom of the water molecules
    n: COORDINATION GROUPA={options['N_WATERS']*3+1} GROUPB=water_group SWITCH={{Q REF={options['R_0']} BETA=-21.497624558253246 LAMBDA=1 R_0={options['R_0']}}}
    t: MATHEVAL ARG=n FUNC={options['N_WATERS']}-x PERIODIC=NO

    r: RESTRAINT ARG=t KAPPA={options['KAPPA']} AT={options['AT']} # apply a harmonic restraint at CN=AT with force constant = KAPPA kJ/mol/CN^2

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_plumed_decoordination(options, filename='plumed.dat'):
    '''
    Write plumed input file for umbrella sampling simulation biasing in the all-molecile coordination number

    Parameters
    ----------
    options : dict
        Options for the Plumed inputs. Required keys are 'ndx', 'ion_group', 'not_ion_group', 'R_0', 'a', 'n_group', 'KAPPA', 'AT', 'STRIDE', and 'FILE'.
        'ndx', 'ion_group', and 'not_ion_group' are the name of the Gromacs-style index file, name of group for the biased ion in the index file, and the name of the group of all other molecules.
        'R_0' is the hydration shell cutoff in nm and 'a' is the switching function parameter with more detail at https://www.plumed.org/doc-v2.9/user-doc/html/switchingfunction.html and https://doi.org/10.1021/jp207018j
        'KAPPA' and 'AT' are the force constant and the umbrella center with more detail at https://www.plumed.org/doc-v2.9/user-doc/html/_r_e_s_t_r_a_i_n_t.html
        'STRIDE' and 'FILE' are the output options for the COLVAR file, frequency of output and name of the output file.
    filename : str
        Name of the Plumed input file, default='plumed.dat'

    Returns
    -------
    filename : str
        Name of the Plumed input file
    
    '''

    f = dedent(f'''\
    ion: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['ion_group']}
    not_ion: GROUP NDX_FILE={options['ndx']} NDX_GROUP={options['not_ion_group']}
    n: COORDINATION GROUPA=ion GROUPB=not_ion SWITCH={{Q REF={options['R_0']} BETA=-{options['a']} LAMBDA=1 R_0={options['R_0']}}}
    t: MATHEVAL ARG=n FUNC={options['n_group']}-x PERIODIC=NO

    r: RESTRAINT ARG=t KAPPA={options['KAPPA']} AT={options['AT']} # apply a harmonic restraint at CN=AT with force constant = KAPPA kJ/mol

    PRINT STRIDE={options['STRIDE']} ARG=* FILE={options['FILE']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename


def write_sbatch_umbrella(options, filename='submit.job'):
    '''
    Write SLURM submission script to run an individual umbrella simulation
    
    Parameters
    ----------
    options : dict
        Options for the SBATCH inputs. Required keys are 'ntasks', 'time', 'job','sim_num', 'gro', 'mdp', and 'top'.
        'ntasks', 'time', and 'job' are SBATCH inputs for the number of CPUs, walltime, and job name.
        'sim_num' is the number of the umbrella simulation.
        'gro', 'mdp', and 'top' are the names of the Gromacs input files with those extensions.
    filename : str
        Name of the SLURM submission script, default='submit.job'

    Returns
    -------
    filename : str
        Name of the SLURM submission script, default='submit.job'

    '''

    f = dedent(f'''\
    #!/bin/bash
    #SBATCH -A chm230020p
    #SBATCH -N 1
    #SBATCH --ntasks-per-node={options['ntasks']}
    #SBATCH -t {options['time']}
    #SBATCH -p RM-shared
    #SBATCH -J '{options['job']}_{options['sim_num']}'
    #SBATCH -o '%x.out'

    module load gcc
    module load openmpi/4.0.2-gcc8.3.1

    source /jet/home/schwinns/.bashrc
    source /jet/home/schwinns/pkgs/gromacs-plumed/bin/GMXRC

    # run umbrella sampling simulations and analysis
    python run_umbrella_sim.py -N {options['sim_num']} -g {options['gro']} -m {options['mdp']} -p {options['top']} -n {options['ntasks']}
    ''')

    out = open(filename, 'w')
    out.write(f)
    out.close()

    return filename