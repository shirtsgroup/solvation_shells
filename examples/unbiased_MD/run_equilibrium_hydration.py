# Run standard MD analysis for solvation shells

import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from solvation_shells_utils import *

plt.rcParams['font.size'] = 12


if __name__ == '__main__':

    # some inputs
    tpr = 'mda_readable.tpr'
    xtc = 'prod.xtc'
    
    cation = 'resname NA'
    anion = 'resname CL'

    cation_label = 'Na$^+$'
    anion_label = 'Cl$^-$'

    get_RDFs = True
    get_shell_probs = False
    get_coordination_number = False

    # intialize and calculate cutoffs
    eq = EquilibriumAnalysis(tpr, xtc, cation=cation, anion=anion)
    print(eq)
    eq.initialize_Solutes()

    # Generate RDFs for cation-water, anion-water, water-water, and cation-anion
    if get_RDFs:
        print('Calculating RDFs...')
        rdfs = eq.generate_rdfs(step=1, filename='rdfs.dat')

    # Calculate the fraction of shells present in the simulation
    if get_shell_probs:
        print('Calculating shell probabilities...')
        eq.shell_probabilities()
        fig, ax = plt.subplots(1,1, figsize=(8,3.55))
        df = eq.cation_shells.merge(eq.anion_shells, on='shell', how='outer')
        df.plot(x='shell', y=['count_x', 'count_y'], kind='bar', legend=False, ax=ax)
        plt.legend([cation_label, anion_label])
        plt.ylabel('probability')
        plt.savefig('shell_probabilities.png')

        fig, ax = plt.subplots(1,2, figsize=(8,3.55))
        df = eq.solute_ci.speciation.speciation_fraction
        sns.barplot(data=df, x='water', y='count', hue='coion', palette='bright', ax=ax[0])
        ax[0].set_xlabel('# of waters in shell')
        ax[0].set_ylabel('probability')
        ax[0].legend(title=f'# of {anion_label} in shell')
        ax[0].set_title(f'{cation_label} hydration shells')

        df = eq.solute_ai.speciation.speciation_fraction
        sns.barplot(data=df, x='water', y='count', hue='coion', palette='bright', ax=ax[1])
        ax[1].set_xlabel('# of waters in shell')
        ax[1].set_ylabel(None)
        ax[1].legend(title=f'# of {cation_label} in shell')
        ax[1].set_title(f'{anion_label} hydration shells')
        plt.savefig('hydration_shell_distributions.png')


    # Calculate the coordination numbers as a function of time and then average
    if get_coordination_number:
        print(f'Average coordination numbers (cation, anion): {eq.get_coordination_numbers()}')
        fig, ax = plt.subplots(1,1, figsize=(8,3.55))
        bin_width = 0.25

        bins = np.arange(eq.coordination_numbers[0,:].min(), eq.coordination_numbers[0,:].max() + bin_width, bin_width)
        plt.hist(eq.coordination_numbers[0,:], label=cation_label, alpha=0.5, bins=bins, edgecolor='k')
        plt.axvline(eq.coordination_numbers.mean(axis=1)[0], ls='dashed', c='tab:blue')

        bins = np.arange(eq.coordination_numbers[1,:].min(), eq.coordination_numbers[1,:].max() + bin_width, bin_width)
        plt.hist(eq.coordination_numbers[1,:], label=anion_label, alpha=0.5, bins=bins, edgecolor='k')
        plt.axvline(eq.coordination_numbers.mean(axis=1)[1], ls='dashed', c='tab:orange')

        plt.xlabel('coordination number')
        plt.ylabel('counts')
        plt.legend()
        plt.savefig('coordination_numbers.png')