# Script to run the polyhedron size analysis on umbrella simulations

import matplotlib.pyplot as plt
from solvation_shells_utils import *
import pickle
import time

if __name__ == '__main__':

    # some inputs
    tpr = 'mda_readable.tpr'
    xtc = 'prod.xtc'
    ci = 'NA'
    ai = 'CL'
    step = 1
    n = 32

    # intialize EquilibriumAnalysis
    print('Loading data...')
    start_time = time.perf_counter()
    eq = EquilibriumAnalysis(tpr, xtc, cation=f'resname {ci}', anion=f'resname {ai}')
    print(f'\t{eq.universe} and {eq.universe.trajectory}')

    # initialize solute
    # eq.initialize_Solutes(step=step)
    load_data_time = time.perf_counter()

    # get polyhedra sizes for cation
    print(f'\nRunning polyhedron size analysis on cations with step={step}...')
    res1 = eq.polyhedron_size(ion='cation', step=step, njobs=n, r0=3.15)
    with open(f'polyhedrons_Na.pl', 'wb') as output:
            pickle.dump(res1, output, pickle.HIGHEST_PROTOCOL)

    # get polyhedra sizes for anion
    print(f'\nRunning polyhedron size analysis on anions with step={step}...')
    res2 = eq.polyhedron_size(ion='anion', step=step, njobs=n, r0=3.95)
    with open(f'polyhedrons_Cl.pl', 'wb') as output:
            pickle.dump(res2, output, pickle.HIGHEST_PROTOCOL)
    
    run_time = time.perf_counter()

    print('\n' + '-'*20 + ' Timing ' + '-'*20)
    print(f'Loading data:       \t\t{load_data_time - start_time:.4f} s')
    print(f'Calculating results:\t\t{run_time - load_data_time:.4f} s\n')
    print(f'Total:              \t\t{run_time - start_time:.4f} s')
    print('-'*48)
