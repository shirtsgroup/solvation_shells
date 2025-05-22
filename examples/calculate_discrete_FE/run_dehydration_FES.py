# Script to run UmbrellaAnalysis to get a FES

from solvation_shells_utils import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    n_sims = 16
    umb_centers = np.linspace(1,9,n_sims)

    print('Loading in data...')
    umb = UmbrellaAnalysis(n_sims)
    bins, fes = umb.calculate_FES(umb_centers, n_bootstraps=0, filename='fes.dat')

    fig, ax = plt.subplots(1,1)
    plt.plot(bins, fes)
    plt.fill_between(bins, fes-umb.error, fes+umb.error, alpha=0.5, facecolor='tab:blue')
    plt.xlabel('Coordination number')
    plt.ylabel('Free energy (kJ/mol)')
    plt.savefig('fes.png')

    performance = np.zeros(n_sims)

    plt.figure(figsize=(12,4))
        
    for i in range(n_sims):
        performance[i] = umb.colvars[i].get_performance(f'prod_{i}.log')
        plt.hist(umb.colvars[i].coordination_number, alpha=0.5, bins=100)

    print(f'Performance = {performance.mean()} ns/day')
    plt.xticks(np.arange(20))
    plt.xlim(umb_centers[0], umb_centers[-1])
    plt.xlabel('Coordination number')
    plt.ylabel('Counts')
    plt.savefig('sampling.png')

    umb.show_overlap()
    plt.savefig('overlap.png')

    umb.find_minima(method='spline_roots', plot=False)
    print(f'Minimum is at {umb.minima_locs[umb.minima_vals.argmin()]}')
