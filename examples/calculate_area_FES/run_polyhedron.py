# Script to run the polyhedron size analysis on umbrella simulations

import matplotlib.pyplot as plt
import pickle
import time

from umbrella_analysis import UmbrellaAnalysis

if __name__ == '__main__':

    # some inputs
    n_sims = 16
    file_path = './'
    tpr = 'mda_readable.tpr'
    njobs = 16
    ci = 'NA'
    ai = 'CL'
    bi = 'NA'
    r0 = 3.15

    # intialize UmbrellaAnalysis
    print('Loading data...')
    start_time = time.perf_counter()
    umb = UmbrellaAnalysis(n_sims)

    # create a universe with trajectory data
    xtcs = [f'prod_{i}.xtc' for i in range(n_sims)]
    umb.create_Universe(tpr, xtcs, cation=f'resname {ci}', anion=f'resname {ai}')
    biased_ion = umb.universe.select_atoms(f'resname {bi}')[0]
    load_data_time = time.perf_counter()

    print('Running polyhedron size analysis...')
    res = umb.polyhedron_size(biased_ion, r0=r0, njobs=njobs)
    with open(f'polyhedrons.pl', 'wb') as output:
            pickle.dump(res, output, pickle.HIGHEST_PROTOCOL)
    
    run_time = time.perf_counter()

    print('\n' + '-'*20 + ' Timing ' + '-'*20)
    print(f'Loading data:       \t\t{load_data_time - start_time:.4f} s')
    print(f'Calculating results:\t\t{run_time - load_data_time:.4f} s')
    print(f'Total:              \t\t{run_time - start_time:.4f} s')
    print('-'*48)
