# UmbrellaAnalysis class to perform analysis on umbrella simulations biased in coordination number

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis.base import Results

import multiprocessing
from multiprocessing import Pool
from functools import partial

from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

import pymbar
from pymbar import timeseries
from solvation_analysis.solute import Solute

from utils.linear_algebra import *
from utils.file_rw import vdW_radii


class UmbrellaSim:

    def __init__(self, COLVAR_file='COLVAR', start=10, stop=-1, by=100):
        '''
        Initialize an umbrella simulation object with a COLVAR file

        Parameters
        ----------
        COLVAR_file : str
            Pattern for the COLVAR files from plumed, default=COLVAR
        start : int
            Index of the first coordination number to read, default=10
        stop : int 
            Index of the last coordination number to read, default=-1
        by : int
            Step by which to read COLVAR entries, default=100

        '''
        
        tmp = np.loadtxt(COLVAR_file, comments='#')

        if tmp.shape[1] == 5: # for a single restrain
            cols = ['time', 'n', 't', 'r.bias', 'r.force2']
            self.data = pd.DataFrame(tmp[start:stop:by,:], columns=cols)
            self.time = self.data.time.to_numpy()
            self.coordination_number = self.data.t.to_numpy()
            self.bias = self.data['r.bias'].to_numpy()
            self.force = self.data['r.force2'].to_numpy()
        elif tmp.shape[1] == 9: # for 2 restraints in water and ion coordination
            cols = ['time', 'n1', 't1', 'r1.bias', 'r1.force2', 'n2', 't2', 'r2.bias', 'r2.force2']
            self.data = pd.DataFrame(tmp[start:stop:by,:], columns=cols)
            self.time = self.data.time.to_numpy()
            self.coordination_number = self.data.t1.to_numpy()
            self.ion_coordination_number = self.data.t2.to_numpy()
            self.water_bias = self.data['r1.bias'].to_numpy()
            self.ion_bias = self.data['r2.bias'].to_numpy()
        else:
            raise ValueError(f'Cannot read file {COLVAR_file}')


    def get_coordination_numbers(self, biased_ion, group, radius, step=1):
        '''
        Calculate the discrete water coordination number as a function of time for biased ion.
        
        Parameters
        ----------
        biased_ion : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the biased ion
        group : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the group to calculate the coordination numbers for (e.g. waters, cations, anions)
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        step : int
            Trajectory step for which to calculate coordination numbers
        
        Returns
        -------
        discrete_coordination_numbers : np.array
            Discrete coordination numbers over the trajectory
        
        '''

        if self.universe is None:
            raise NameError('No universe data found. Try `create_Universe()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # initialize coordination number as a function of time
        self.discrete_coordination_numbers = np.zeros(len(self.universe.trajectory[::step]))

        for i,ts in enumerate(self.universe.trajectory[::step]):
            d = distances.distance_array(ion, group, box=ts.dimensions)
            self.discrete_coordination_numbers[i] = (d <= radius).sum()

        return self.discrete_coordination_numbers


    def get_performance(self, log_file='prod.log'):
        '''
        Get the performance of this umbrella simulation from the log file
        
        Parameters
        ----------
        log_file : str
            Name of the log file for the umbrella simulation, default=prod.log

        Returns
        -------
        performance : float
            Performance for the simulation in ns/day

        '''

        f = open(log_file)
        lines = f.readlines()
        tmp = [float(line.split()[1]) for line in lines if line.startswith('Performance')]
        self.performance = tmp[0]

        return self.performance
    

    def create_Universe(self, top, traj=None, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Create an MDAnalysis Universe for the individual umbrella simulation.

        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc)
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'

        Returns
        -------
        universe : MDAnalysis.Universe object
            MDAnalysis Universe with the toplogy and coordinates for this umbrella

        '''

        self.universe = mda.Universe(top, traj)    

        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')
    
        return self.universe


    def initialize_Solute(self, ion, cutoff, step=1):
        '''
        Initialize the Solute object from SolvationAnalysis for the ion. Saves the solute
        in attribute `solute`. 
        
        Parameters
        ----------
        ion : MDAnalysis.AtomGroup or str
            Ion to create a Solute object for, if a str should be MDAnalysis selection language
        cutoff : float
            Hydration shell cutoff in Angstroms
        step : int
            Trajectory step for which to run the Solute

        Returns
        -------
        solute : solvation_analysis.solute.Solute
            SolvationAnalysis Solute object for `ion` with hydration shell `cutoff`
            
        '''

        if isinstance(ion, str): # if provided selection language, make AtomGroup
            g = self.universe.select_atoms(ion)
        else: # else assume input is AtomGroup
            g = ion

        if g[0].charge > 0:
            other_ions = self.cations - g
            coions = self.anions
            name = 'cation'
        elif g[0].charge < 0:
            other_ions = self.anions - g
            coions = self.cations
            name = 'anion'
        else:
            raise TypeError('Your ion is not charged, and so not an ion.')

        
        self.solute = Solute.from_atoms(g, {'water' : self.waters, 'ion' : other_ions, 'coion' : coions}, 
                                        solute_name=name, radii={'water' : cutoff, 'ion' : cutoff, 'coion' : cutoff})
        self.solute.run(step=step)

        return self.solute
    

class UmbrellaAnalysis:

    def __init__(self, n_umbrellas, COLVAR_file='COLVAR_', start=10, stop=-1, by=100, T=300, verbose=True):
        '''
        Initialize the umbrella sampling analysis object with collective variable files for each simulation
        
        Parameters
        ----------
        n_umbrellas : int
            Number of umbrella simulations
        COLVAR_file : str
            Pattern for the COLVAR files from plumed, default=COLVAR_
        start : int
            Index of the first coordination number to read, default=10
        stop : int
            Index of the last coordination number to read, default=-1
        by : int
            Step by which to read COLVAR entries, default=100
        T : float
            Temperature (K), default=300
        verbose : bool
            Verbosity, controls whether to print detailed information and progress bars, default=True

        '''

        # initialize some variables
        self.kB = 1.380649 * 10**-23 * 10**-3 * 6.022*10**23 # Boltzmann (kJ / K)
        self.kT = self.kB*T
        self.beta = 1/self.kT
        self._fes = None
        self.universe = None
        self.coordination_numbers = None
        self.polyhedron_sizes = None
        self.verbose = verbose

        # read in collective variable files
        self.colvars = []

        for i in range(n_umbrellas):
            filename = f'{COLVAR_file}{i}'
            self.colvars.append(UmbrellaSim(filename, start=start, stop=stop, by=by))

        self.vdW_radii = vdW_radii().get_dict() # I am sure there is a better way to do this... but I am not taking the time now

    
    def __repr__(self):
        if self.universe is not None:
            return f'UmbrellaAnalysis object with {len(self.colvars)} simulations and {len(self.universe.trajectory)} frames'
        else:
            return f'UmbrellaAnalysis object with {len(self.colvars)} simulations'

        
    def calculate_FES(self, CN0_k, KAPPA=100, n_bootstraps=0, nbins=200, d_min=2, d_max=8, bw=0.02, error=True, mintozero=True, filename=None):
        '''
        Calculate the free energy surface with pymbar
        
        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers
        KAPPA : float, array-like
            Strength of the harmonic potential (kJ/mol/CN^2), default=100
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        nbins : int
            Number of bins for the free energy surface
        d_min : float
            Minimum coordination number for the free energy surface
        d_max : float
            Maximum coordination number for the free energy surface
        bw : float
            Bandwidth for the KDE
        error : bool
            Calculate error. If True and n_bootstraps > 0, then will calculate the bootstrapped error. 
            Otherwise, calculates the analytical histogram error, default=True
        mintozero : bool
            Shift the minimum of the free energy surface to 0
        filename : str
            Name of the file to save the free energy surface, default=None

        Returns
        -------
        bin_centers : np.array
            Coordination numbers for the FES
        fes : np.array
            FES along the coordination number in kJ/mol

        '''

        # Step 1: Subsample timeseries
        print('Subsampling timeseries...')
        u_kn, u_kln, N_k, d_kn = self._subsample_timeseries(error=error, plot=True)
        
        # Step 2: Bin the data
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(d_min, d_max, nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        # Step 3: Evaluate reduced energies in all umbrellas
        print('Evaluating energies...')
        u_kln = self._evaluate_reduced_energies(CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA)

        # Step 4: Compute and output the FES
        print('Calculating the free energy surface...')
        fes = pymbar.FES(u_kln, N_k, verbose=False)
        d_n = pymbar.utils.kn_to_n(d_kn, N_k=N_k)
        if not error:
            fes.generate_fes(u_kn, d_n, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method=None)
            results['df_i'] = np.zeros(len(results['f_i']))
        elif n_bootstraps == 0:
            fes.generate_fes(u_kn, d_n, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='analytical')
        else:
            fes.generate_fes(u_kn, d_n, fes_type='kde', kde_parameters=kde_params, n_bootstraps=n_bootstraps)
            results = fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='bootstrap')

        if mintozero:
            results['f_i'] = results['f_i'] - results['f_i'].min()

        # Step 5: Save FES information in the object
        print('Saving results...')
        self.umbrella_centers = CN0_k
        self._u_kln = u_kln
        self.u_kn = u_kn
        self._N_k = N_k
        self._fes = fes                     # underlying pymbar.FES object
        self._results = results             # underlying results object
        self.bin_centers = bin_center_i
        self.fes = results['f_i']*self.kT
        self.error = results['df_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([self.bin_centers, self.fes, self.error]).T, header='coordination number, free energy (kJ/mol), error (kJ/mol)')

        return self.bin_centers, self.fes
    

    def calculate_discrete_FE(self, biased_ion, radius, n_bootstraps=0, cn_range=None, filename=None, **kwargs):
        '''
        Calculate the free energies associated with the discrete coordination number states from the continuous coordination number simulations

        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        cn_range : array-like
            Coordination number range to calculate the discrete free energoes, default=None means use the min and max observed
        filename : str
            Name of the file to save the discrete free energies

        Returns
        -------
        results : MDAnalysis Results class with attributes `coordination_number`, `free_energy`, and `error`
            Free energies for the discrete coordination numbers. If `n_bootstraps` is 0, all errors will be 0.

        '''

        if self._fes is None:
            raise ValueError('Continuous coordination number free energy surface not found. Try `calculate_FES()` first')
        
        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # determine indices to remove to ensure COLVAR time and Universe time match
        n_sims = len(self.colvars)
        total_frames = self.universe.trajectory.n_frames
        umb_frames = self.colvars[0].time.shape[0]
        to_remove = np.arange(umb_frames, total_frames+1, umb_frames+1)

        if self.coordination_numbers is None:
            cn = self.get_coordination_numbers(ion, radius, filename='tmp_CN.csv', **kwargs)
        else:
            cn = self.coordination_numbers
        
        cn = np.delete(cn, to_remove)
        if cn_range is None:
            cn_range = (cn.min(), cn.max())
        print(f'\tDiscrete coordination numbers range: ({cn.min()}, {cn.max()})')

        # prepare the Results object
        results = Results()
        results.coordination_number = np.arange(cn_range[0], cn_range[1]+1)
        results.free_energy = np.zeros((cn_range[1] - cn_range[0] + 1))
        results.error = np.zeros((cn_range[1] - cn_range[0] + 1))

        # get the discrete bins
        bin_edges = np.arange(cn_range[0]-0.5, cn_range[1]+1.5)
        bins = np.arange(cn_range[0], cn_range[1]+1)

        if n_bootstraps > 0:
            print(f'Calculating discrete free energies with {n_bootstraps} bootstraps...')
            # if calculating error, get uncorrelated discrete coordination numbers
            N_k = self._N_k
            cn_kn = cn.reshape((n_sims, umb_frames))
            for k in range(n_sims):
                idx = self.uncorrelated_indices[k]
                cn_kn[k, 0:N_k[k]] = cn_kn[k, idx]

            cn = pymbar.utils.kn_to_n(cn_kn, N_k=N_k)

            self._fes.generate_fes(self.u_kn, cn, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges}, n_bootstraps=n_bootstraps)
            res = self._fes.get_fes(bins, reference_point='from-lowest', uncertainty_method='bootstrap')
            results.error = res['df_i']*self.kT

        else:
            print(f'Calculating discrete free energies without error...')
            # do not calculate error, since unsure what histogram error means for this case
            self._fes.generate_fes(self.u_kn, cn, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges})
            res = self._fes.get_fes(bins, reference_point='from-lowest', uncertainty_method=None)

        # convert to kJ/mol and save in Results object
        results.free_energy = res['f_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([results.coordination_number, results.free_energy, results.error]).T, header='coordination number, free energy (kJ/mol), error (kJ/mol)')

        return results
    

    def calculate_area_FES(self, area_range=None, nbins=50, n_bootstraps=0, filename=None):
        '''
        Calculate the free energy surfaces in the coordination shell cross-sectional areas collective variable space

        Parameters
        ----------
        area_range : array-like, shape (2,)
            Min and max area values to calculate the FES, default=None means use the minimum and maximum areas from the timeseries
        nbins : int
            Number of bins for the FES histogram, default=50
        n_bootstraps : int
            Number of bootstraps for the uncertainty calculation, default=0
        filename : str
            Name of the file to save the FES in area, default=None means do not save

        Returns
        -------
        results : MDAnalysis Results class with attributes `coordination_number`, `free_energy`, and `error`
            Free energies for the discrete coordination numbers. If `n_bootstraps` is 0, all errors will be 0.

        '''

        if self._fes is None:
            raise ValueError('Continuous coordination number free energy surface not found. Try `calculate_FES()` first')
        
        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')
        
        if self.polyhedron_sizes is None:
            raise ValueError('No polyhedron size data. Try  `polyhedron_size()` first')

        # load in polyhedrons and remove extra frames
        n_sims = len(self.colvars)
        total_frames = self.universe.trajectory.n_frames
        umb_frames = self.colvars[0].time.shape[0]
        to_remove = np.arange(umb_frames, total_frames+1, umb_frames+1)

        poly = self.polyhedron_sizes
        area = np.delete(poly.areas, to_remove)

        # get uncorrelated areas
        N_k = self._N_k
        area_kn = area.reshape((n_sims, umb_frames))
        for k in range(n_sims):
            idx = self.uncorrelated_indices[k]
            area_kn[k, 0:N_k[k]] = area_kn[k, idx]

        area = pymbar.utils.kn_to_n(area_kn, N_k=N_k)

        if area_range is None:
            area_range = (area.min(), area.max())

        # bin the areas for the FES
        bin_center_i = np.zeros([nbins])
        bin_edges = np.linspace(area_range[0], area_range[1], nbins + 1)
        for i in range(nbins):
            bin_center_i[i] = 0.5 * (bin_edges[i] + bin_edges[i + 1])

        # generate the FES in area
        self._fes.generate_fes(self.u_kn, area, fes_type='histogram', histogram_parameters={'bin_edges' : bin_edges}, n_bootstraps=n_bootstraps)
        res = self._fes.get_fes(bin_center_i, reference_point='from-lowest', uncertainty_method='bootstrap')
        res['f_i'] = res['f_i']*self.kT
        res['df_i'] = res['df_i']*self.kT

        if filename is not None:
            np.savetxt(filename, np.vstack([bin_center_i, res['f_i'], res['df_i']]).T, header='max polyhedron area (Angstroms^2), free energy (kJ/mol), error (kJ/mol)')

        return bin_center_i, res['f_i'], res['df_i']
    

    def show_overlap(self):
        '''
        Compute the overlap matrix and plot as a heatmap
        
        Returns
        -------
        heatmap : sns.Axes
            Heatmap of overlap from seaborn
        
        '''
        
        overlap = self._fes.mbar.compute_overlap()

        df = pd.DataFrame(overlap['matrix'], columns=[i for i in range(len(self.colvars))])
        fig, ax = plt.subplots(1,1, figsize=(10,8))
        heatmap = sns.heatmap(df, annot=True, fmt='.2f', ax=ax)

        return heatmap
    

    def average_coordination_number(self, CN0_k=None, KAPPA=100):
        '''
        Compute the average coordination number with a Boltzmann-weighted average
        
        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers, default=None because it is not
            necessary if there is already an underlying MBAR object
        KAPPA : float
            Strength of the harmonic potential (kJ/mol/nm^2), default=100

        Returns
        -------
        results['mu'] : float
            Boltzmann-weighted average coordination number
        results['sigma'] : float
            Standard deviation of the mean coordination number

        '''

        # first, subsample the timeseries to get d_kn (the uncorrelated coordination numbers)
        u_kn, u_kln, N_k, d_kn = self._subsample_timeseries()

        if self._fes is None: # if no underlying MBAR object, create one
            u_kln = self._evaluate_reduced_energies(CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA)
            mbar = pymbar.MBAR(u_kln, N_k)

        else: # otherwise get it from FES
            mbar = self._fes.get_mbar()

        results = mbar.compute_expectations(d_kn)

        return results['mu'], results['sigma']

        
    def find_minima(self, plot=False, method='find_peaks', **kwargs):
        '''
        Find the local minima of the free energy surface. `method` options are 'find_peaks'
        and 'spline_roots'. 'find_peaks' uses scipy.signal find_peaks to locate the minima
        based on peak properties. 'spline_roots' fits a UnivariateSpline to the FES and finds
        its minima by solving df/dx=0. 

        Parameters
        ----------
        plot : bool
            Whether to plot the minima on the free energy surface, default=False
        method : str
            Method to use to locate the minima, default='find_peaks'
        
        Returns
        -------
        minima_loc : np.array
            Bin locations of the minima in the FES

        '''

        if method == 'find_peaks':

            peaks,_ = find_peaks(-self.fes, **kwargs)
            self.minima_idx = peaks
            self.minima_locs = self.bin_centers[peaks]
            self.minima_vals = self.fes[peaks]

        elif method == 'spline_roots':

            self.spline = self._fit_spline(**kwargs)
            self.minima_locs, self.minima_vals = self._get_spline_minima()

        if plot:
            plt.plot(self.bin_centers, self.fes)
            plt.scatter(self.minima_locs, self.minima_vals, marker='x', c='r')
            plt.xlabel('Coordination number')
            plt.ylabel('Free energy (kJ/mol)')

        return self.minima_locs
    
    
    def get_dehydration_energy(self, cn1, cn2, uncertainty_method=None):
        '''
        Calculate the dehydration energy from cn1 to cn2. This function fits a spline to the free energy surface
        and estimates the energies as the spline evaluated at cn1 and cn2. For positive free energy, corresponding to
        how much free energy is needed to strip a coordinated water, cn1 should be the higher energy coordination state.

        Parameters
        ----------
        cn1 : float
            Coordination number of state 1 to calculate dG = G_1 - G_2
        cn2 : float
            Coordination number of state 2 to calculate dG = G_1 - G_2
        uncertainty_method : str
            Method to calculate the uncertainty. Currently, the only method available is 'bootstrap'. Default=None means
            it will not calculate uncertainty.

        Returns
        -------
        dG : float
            Free energy difference between cn1 and cn2
        dG_std : float
            Standard deviation in the free energy difference, only returned if uncertainty_method='bootstrap'
        
        '''

        if uncertainty_method == 'bootstrap':
            n_bootstraps = len(self._fes.kdes)
            x = self.bin_centers.reshape(-1,1)

            dG_boots = np.zeros(n_bootstraps)
            for b in range(n_bootstraps):
                fes_boot = -self._fes.kdes[b].score_samples(x)*self.kT
                spline = self._fit_spline(self.bin_centers, fes_boot)
                dG_boots[b] = spline(cn1) - spline(cn2)

            return dG_boots.mean(), dG_boots.std()
        
        else:
            spline = self._fit_spline(self.bin_centers, self.fes)
            dG = spline(cn1) - spline(cn2)

            return dG
        

    def rdfs_by_coordination(self, biased_ion, CN_range, bin_width=0.05, range=(0,20)):
        '''
        Calculate radial distribution functions as a function of the biased coordination number. This method 
        calculates the RDFs for ion-water, ion-ion, and ion-coion using MDAnalysis InterRDF. It saves 
        the data in a dictionary attribute `rdfs` with keys 'i-w', 'i-i', 'i-ci'. Each key corresponds 
        to a dictionary of coordination numbers. 
        
        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        CN_range : array-like
            Range of coordination numbers to calculate the RDF for
        bin_width : float
            Width of the bins for the RDFs, default=0.05
        range : array-like
            Range over which to calculate the RDF, default=(0,20)

        Returns
        -------
        rdfs : dict
            Dictionary of dictionaries with all the results from InterRDF
        
        '''

        if self.coordination_numbers is None:
            raise ValueError('Discrete coordination number data not found. Try `get_coordination_numbers()` first')
        
        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # decide which ions are the same as the biased ion
        if ion in self.cations:
            ions = self.cations - ion
            coions = self.anions - ion
        elif ion in self.anions:
            ions = self.anions - ion
            coions = self.cations - ion


        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {
            'i-w'  : {},
            'i-i'  : {},
            'i-ci' : {}
        }

        for CN in CN_range:
            idx = self.coordination_numbers == CN
            print(f'Coordination number {CN}: {idx.sum()} frames')

            if idx.sum() > 0:
                i_w = rdf.InterRDF(ion, self.waters, nbins=nbins, range=range, norm='rdf')
                i_w.run(frames=idx)
                self.rdfs['i-w'][CN] = i_w.results

                i_i = rdf.InterRDF(ion, ions, nbins=nbins, range=range, norm='rdf')
                i_i.run(frames=idx)
                self.rdfs['i-i'][CN] = i_i.results

                i_ci = rdf.InterRDF(ion, coions, nbins=nbins, range=range, norm='rdf')
                i_ci.run(frames=idx)
                self.rdfs['i-ci'][CN] = i_ci.results

        return self.rdfs


    def angular_distributions_by_coordination(self, biased_ion, CN_range, bin_width=0.05, range=(1,10)):
        '''
        Calculate water angular distributions as a function of the biased coordination number. This method
        saves the data in a dictionary attribute `angular_distributions` with keys 'theta' and 'phi'. 
        
        Parameters
        ----------
        biased_ion : MDAnalysis.Atom
            An MDAnalysis Atom of the biased ion
        CN_range : array-like
            Range of coordination numbers to calculate the distributions for
        bin_width : float
            Width of the bins in the r direction, default=0.05
        range : array-like
            Radial range over which to calculate the distributions, default=(1,10)

        Returns
        -------
        angular_distributions : dict
            Dictionary of dictionaries with all the results
        
        '''

        if self.coordination_numbers is None:
            raise ValueError('Discrete coordination number data not found. Try `get_coordination_numbers()` first')

        nbins = int((range[1] - range[0]) / bin_width)
        rbins = np.linspace(range[0], range[1], nbins)
        thbins = np.linspace(0,180, nbins)
        phbins = np.linspace(-180,180, nbins)

        self.angular_distributions = {
            'theta' : {},
            'phi' : {}
        }

        for CN in CN_range:
            th_hist,th_x,th_y = np.histogram2d([], [], bins=[rbins,thbins])
            ph_hist,ph_x,ph_y = np.histogram2d([], [], bins=[rbins,phbins])

            idx = self.coordination_numbers == CN
            print(f'Coordination number {CN}: {idx.sum()} frames')

            if idx.sum() > 0:
                for i, ts in enumerate(self.universe.trajectory[idx]):
                    d = distances.distance_array(mda.AtomGroup([biased_ion]), self.waters, box=ts.dimensions)
                    closest_water = self.waters[d.argmin()]
                    self.universe.atoms.translate(-biased_ion.position) # set the ion as the origin
                    my_waters = self.waters.select_atoms(f'(point 0 0 0 {range[1]}) and (not index {closest_water.index})', updating=True) # select only waters near the ion

                    if len(my_waters) > 0:
                        # rotate system so z axis is oriented with ion-closest water vector
                        v2 = np.array([0,0,1])
                        rotation_matrix = get_rotation_matrix(closest_water.position, v2)
                        positions = rotation_matrix.dot(my_waters.positions.T).T

                        # convert to spherical coordinates, centered at the ion
                        r = np.sqrt(positions[:,0]**2 + positions[:,1]**2 + positions[:,2]**2)
                        th = np.degrees(np.arccos(positions[:,2] / r))
                        ph = np.degrees(np.arctan2(positions[:,1], positions[:,0]))

                        # histogram to get the probability density
                        h1,_,_ = np.histogram2d(r, th, bins=[rbins,thbins])
                        h2,_,_ = np.histogram2d(r, ph, bins=[rbins,phbins])
                        
                        th_hist += h1
                        ph_hist += h2

                th_data = {'r' : th_x, 'theta' : th_y, 'hist' : th_hist.T}
                ph_data = {'r' : ph_x, 'phi' : ph_y, 'hist' : ph_hist.T}
                self.angular_distributions['theta'][CN] = th_data
                self.angular_distributions['phi'][CN] = ph_data

        return self.angular_distributions
    

    def water_dipole_distribution(self, biased_ion, radius, n_max=12, njobs=1, step=1):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        biased_ion : MDAnalysis.Atom
            Ion to calculate the distribution for, the ion whose coordination shell has been biased.
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only
        n_max : int
            Maximum number of coordinated waters, if discrete coordination numbers have been calculated, will use
            the max of `self.coordination_numbers`, default=12
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.
        step : int
            Step to iterate the trajectory when running the analysis, default=1

        Returns
        -------
        results :  MDAnalysis Results class with attribute angles
            Angles for all waters coordinated with biased ion

        '''

        if self.coordination_numbers is not None: # if discrete coordination numbers have been calculated create a time x max number coordinating array
            n_max = self.coordination_numbers.max()

        # prepare the Results object
        results = Results()
        results.angles = np.empty((len(self.universe.trajectory[::step]),n_max))
        results.angles[:] = np.nan # should be NaN if not specified

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                ang = self._water_dipole_per_frame(i, biased_ion, radius=radius)
                results.angles[i,:] = ang

        else: # run in parallel
            
            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._water_dipole_per_frame,
                                    biased_ion=biased_ion,
                                    radius=radius,
                                    n_max=n_max)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            ang = np.asarray(result)
            results.angles = ang 
        
        return results
    

    def polyhedron_size(self, biased_ion, r0=3.15, njobs=1, step=1):
        '''
        Calculate the maximum cross-sectional areas and volumes as time series for coordination shells.
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        biased_ion : str, MDAnalysis.Atom
            Biased ion in the simulation to calculate polyhedrons for
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.

        Returns
        -------
        results : MDAnalysis Results class with attributes `volumes` and `areas`
            Volume and maximum cross-sectional area for the polyhedron
        
        '''

        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')

        # prepare the Results object
        results = Results()
        results.areas = np.zeros(len(self.universe.trajectory[::step]))
        results.volumes = np.zeros(len(self.universe.trajectory[::step]))

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                a,v = self._polyhedron_size_per_frame(i, biased_ion, r0=r0)
                results.areas[i] = a
                results.volumes[i] = v

        else: # run in parallel

            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._polyhedron_size_per_frame,
                                    biased_ion=biased_ion,
                                    r0=r0,
                                    for_visualization=False)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            result = np.asarray(result)
            results.areas = result[:,0]
            results.volumes = result[:,1]

        self.polyhedron_sizes = results

        return results


    def ion_pairing(self, biased_ion, ion_pair_cutoffs, plot=False, njobs=1):
        '''
        Calculate the frequency of ion pairing events as defined in https://doi.org/10.1063/1.4901927 
        over the umbrella trajectories. This method saves the time series of the ion pairing states for
        the biased ion and returns the frequency distribution.

        Parameters
        ----------
        biased_ion : str or MDAnalysis.AtomGroup
            Biased ion in the simulation
        ion_pair_cutoffs : dict or MDAnalysis.analysis.base.Results of tuples
            Dictionary with keys ['CIP', 'SIP', 'DSIP', 'FI'] with values (min, max) for each region
        plot : bool
            Whether to plot the distribution, default=False
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use MDAnalysis
            OpenMP backend to calculate distances.
        
        Returns
        -------
        freq : pandas.DataFrame
            Distribution of ion pairing frequencies, sums to 1

        '''

        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        # check whether biased ion is cation or anion
        if ion in self.cations:
            coions = self.anions
        elif ion in self.anions:
            coions = self.cations
        else:
            raise ValueError(f'Biased ion {ion} does not belong to anions ({self.anions}) or cations ({self.cations}).')

        if self.universe is None:
            raise ValueError('No underlying MDAnalysis.Universe. Try `create_Universe()` first')

        self.ion_pairs = Results()
        self.ion_pairs['CIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['SIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['DSIP'] = np.zeros(len(self.universe.trajectory))
        self.ion_pairs['FI'] = np.zeros(len(self.universe.trajectory))

        # set backend depending on number of CPUs available
        if njobs == 1:
            backend = 'serial'
        else:
            backend = 'OpenMP'

        # increment the state the biased ion is in
        for i,ts in tqdm(enumerate(self.universe.trajectory)):
            d = distances.distance_array(ion, coions, box=ts.dimensions, backend=backend)[0,:]
            idx, dist = d.argmin(), d.min()
            for ip,range in ion_pair_cutoffs.items():                
                if range[0] <= dist <= range[1]:
                    self.ion_pairs[ip][i] += 1
                    break

        # calculate the distribution from the time series
        df = pd.DataFrame(self.ion_pairs.data)
        freq = pd.DataFrame(df.sum() / len(self.universe.trajectory))

        if plot:
            freq.plot(kind='bar', legend=None)
            plt.ylabel('Frequency')
            plt.savefig('ion_pair_distribution.png')
            plt.show()

        return freq


    def create_Universe(self, top, traj=None, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Create an MDAnalysis Universe for the individual umbrella simulation.

        Parameters
        ----------
        top : str
            Name of the topology file (e.g. tpr, gro, pdb)
        traj : str or list of str
            Name(s) of the trajectory file(s) (e.g. xtc), default=None
        water : str
            MDAnalysis selection language for the water oxygen, default='type OW'
        cation : str
            MDAnalysis selection language for the cation, default='resname NA'
        anion : str
            MDAnalysis selection language for the anion, default='resname CL'

        Returns
        -------
        universe : MDAnalysis.Universe object
            MDAnalysis Universe with the toplogy and coordinates for this umbrella

        '''

        self.universe = mda.Universe(top, traj)    

        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')
    
        return self.universe
    

    def get_coordination_numbers(self, biased_ion, radius, filename=None, njobs=1, verbose=None):
        '''
        Calculate the discrete total coordination number as a function of time for biased ion.
        
        Parameters
        ----------
        biased_ion : str, MDAnalysis.AtomGroup
            Either selection language for the biased ion or an MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)
        filename : str
            Name of the file to save the discrete coordination numbers, should be a csv file, default=None means do not save
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use MDAnalysis
            OpenMP backend to calculate distances.
        verbose : bool
            Whether to show progress bar, default=None means use UmbrellaAnalysis settings
        
        Returns
        -------
        coordination_numbers : np.array
            Discrete coordination numbers over the trajectory
        
        '''

        if self.universe is None:
            raise NameError('No universe data found. Try `create_Universe()` first')
        
        if verbose is None:
            verbose = self.verbose

        # make biased_ion into MDAnalysis AtomGroup
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)
        else:
            ion = biased_ion

        if njobs == 1: # run on 1 CPU

            print(f'\nCalculating the discrete coordination numbers...')

            # initialize coordination number as a function of time
            self.coordination_numbers = np.zeros(len(self.universe.trajectory))

            if verbose:
                for i,ts in tqdm(enumerate(self.universe.trajectory)):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='serial')
            else:
                for i,ts in enumerate(self.universe.trajectory):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='serial')

        else: # run in parallel

            print(f'\nCalculating the discrete coordination numbers using {njobs} CPUs...')

            # initialize coordination number as a function of time
            self.coordination_numbers = np.zeros(len(self.universe.trajectory))

            if verbose:
                for i,ts in tqdm(enumerate(self.universe.trajectory)):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='OpenMP')
            else:
                for i,ts in enumerate(self.universe.trajectory):
                    self.coordination_numbers[i] = self._coordination_number_per_frame(i, ion, radius, backend='OpenMP')

        if filename is not None:
            df = pd.DataFrame()
            df['idx'] = np.arange(0,len(self.coordination_numbers))
            df['coordination_number'] = self.coordination_numbers
            df.to_csv(filename, index=False)

        return self.coordination_numbers


    def _subsample_timeseries(self, error=True, plot=False):
        '''
        Subsample the timeseries to get uncorrelated samples. This function also sets up the variables 
        needed for pymbar.MBAR object and pymbar.FES object.

        Parameters
        ----------
        error : bool
            Calculate error. If False, we do not need to subsample timeseries, default=True
        plot : bool
            Plot the sampling distributions. If both `plot` and `error` are True, then plot the uncorrelated samples, default=False
        
        Returns
        -------
        u_kn : array-like
            u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k, 
            reshaped for uncorrelated samples
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k, shaped properly
        N_k : array-like
            Number of samples frum umbrella k, reshaped for uncorrelated samples
        d_kn : array-like
            d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k, reshaped for uncorrelated samples

        '''
        
        # Step 1a: Setting up
        K = len(self.colvars)                       # number of umbrellas
        N_max = self.colvars[0].time.shape[0]       # number of data points in each timeseries of coordination number
        N_k, g_k = np.zeros(K, int), np.zeros(K)    # number of samples and statistical inefficiency of different simulations
        d_kn = np.zeros([K, N_max])                 # d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k
        u_kn = np.zeros([K, N_max])                 # u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
        self.uncorrelated_samples = []              # Uncorrelated samples of different simulations
        self.uncorrelated_indices = []
        ion_restraint = (self.colvars[0].data.shape[1] == 9) # determine if there is an ion coordination restraint

        # Step 1b: Read in and subsample the timeseries
        for k in range(K):
            # if using 2 restraints, calculate the potential from the ion coordination restraint
            if ion_restraint:
                u_kn[k] = self.beta * 10000/2 * self.colvars[k].ion_coordination_number**2 # KAPPA = 10,0000 and centered at CN = 0

            d_kn[k] = self.colvars[k].coordination_number
            N_k[k] = len(d_kn[k])
            d_temp = d_kn[k, 0:N_k[k]]
            if error:
                g_k[k] = timeseries.statistical_inefficiency(d_temp)     
                print(f"Statistical inefficiency of simulation {k}: {g_k[k]:.3f}")
                indices = timeseries.subsample_correlated_data(d_temp, g=g_k[k]) # indices of the uncorrelated samples
            else:
                indices = np.arange(len(self.colvars[k].coordination_number))
            
            # Update u_kn and d_kn with uncorrelated samples if calculating error
            N_k[k] = len(indices)    # At this point, N_k contains the number of uncorrelated samples for each state k                
            u_kn[k, 0:N_k[k]] = u_kn[k, indices]
            d_kn[k, 0:N_k[k]] = d_kn[k, indices]
            if error:
                self.uncorrelated_samples.append(d_kn[k, indices])
                self.uncorrelated_indices.append(indices)

        N_max = np.max(N_k) # shorten the array size
        u_kln = np.zeros([K, K, N_max]) # u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k

        if error and plot:
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.set_xlabel('Coordination number, continuous')
            ax.set_ylabel('Counts')
            for k in range(K):
                ax.hist(self.uncorrelated_samples[k], bins=50, alpha=0.5)

            ax.set_xlim(d_kn.min()-0.5, d_kn.max()+0.5)
            fig.savefig('uncorrelated_sampling.png')
            plt.close()

        elif plot:
            fig, ax = plt.subplots(1,1, figsize=(8,4))
            ax.set_xlabel('Coordination number, continuous')
            ax.set_ylabel('Counts')
            for k in range(K):
                ax.hist(self.colvars[k].coordination_number, bins=50, alpha=0.5)

            ax.set_xlim(d_kn.min()-0.5, d_kn.max()+0.5)
            fig.savefig('sampling.png')
            plt.close()

        return u_kn, u_kln, N_k, d_kn
    

    def _evaluate_reduced_energies(self, CN0_k, u_kn, u_kln, N_k, d_kn, KAPPA=100):
        '''
        Create the u_kln matrix of reduced energies from the umbrella simulations.

        Parameters
        ----------
        CN0_k : array-like
            Coordination numbers at the umbrella simulation centers
        u_kn : array-like
            u_kn[k,n] is the reduced potential energy without umbrella restraints of snapshot n of umbrella simulation k
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k
        N_k : array-like
            Number of samples frum umbrella k
        d_kn : array-like
            d_kn[k,n] is the coordination number for snapshot n from umbrella simulation k
        KAPPA : float, array-like
            Strength of the harmonic potential (kJ/mol/CN^2), default=100

        Returns
        -------
        u_kln : array-like
            u_kln[k,n] is the reduced potential energy of snapshot n from umbrella simulation k, calculated from
            u_kn and the harmonic restraint, shaped properly

        '''


        K = len(self.colvars)                       # number of umbrellas
        beta_k = np.ones(K) * self.beta             # inverse temperature of simulations (in 1/(kJ/mol)) 

        # spring constant (in kJ/mol/CN^2) for different simulations
        # coerce into a np.array
        if isinstance(KAPPA, (float, int)): 
            K_k = np.ones(K)*KAPPA                   
        elif not isinstance(KAPPA, np.ndarray):
            K_k = np.array(KAPPA)
        else:
            K_k = KAPPA

        for k in range(K):
            for n in range(N_k[k]):
                # Compute the distance from the center of simulation k in coordination number space
                dd = d_kn[k,n] - CN0_k

                # Compute energy of snapshot n from simulation k in umbrella potential l
                u_kln[k,:,n] = u_kn[k,n] + beta_k[k] * (K_k / 2) * dd ** 2

        return u_kln


    def _fit_spline(self, bins=None, fes=None):
        '''
        Fit a scipy.interpolate.UnivariateSpline to the FES. Uses a quartic spline (k=4) 
        and interpolates with all points, no smoothing (s=0)

        Parameters
        ----------
        bins : np.array
            Bins of the FES to fit to spline, default=None means use self.bin_centers
        fes : np.array
            FES to fit to spline, default=None means use self.fes

        Returns
        -------
        f : the interpolated spline

        '''

        from scipy.interpolate import UnivariateSpline

        if bins is None:
            bins = self.bin_centers
        if fes is None:
            fes = self.fes

        f = UnivariateSpline(bins, fes, k=4, s=0)
        
        return f
    

    def _coordination_number_per_frame(self, frame_idx, biased_ion, radius, **kwargs):
        '''
        Calculate the discrete total coordination number as a function of time for biased ion.

        Parameters
        ----------
        frame_idx : int
            Index of the trajectory frame
        biased_ion : MDAnalysis.AtomGroup
            MDAnalysis AtomGroup of the biased ion
        radius : float
            Hydration shell cutoff for the ion (Angstroms)

        Returns
        -------
        coordination_number : int
            Discrete coordination number around the biased ion
        
        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        # calculate distances and compare to hydration shell cutoff
        d = distances.distance_array(biased_ion, self.universe.select_atoms('not type HW* MW') - biased_ion, 
                                     box=self.universe.dimensions, **kwargs)
        coordination_number = (d <= radius).sum()

        return coordination_number
    

    def _water_dipole_per_frame(self, frame_idx, biased_ion, radius, n_max=12):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        biased_ion : MDAnalysis.Atom
            Ion to calculate the distribution for, the ion whose coordination shell has been biased.
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only
        n_max : int
            Maximum number of coordinated waters, if discrete coordination numbers have been calculated, will use
            the max of `self.coordination_numbers`, default=12

        Returns
        -------
        angles : np.array
            Angles for the waters coordinated on this frame, size n_max

        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        my_atoms = self.universe.select_atoms(f'sphzone {radius} index {biased_ion.index}') - biased_ion
        my_waters = my_atoms & self.waters # intersection operator to get the OW from my_atoms

        angles = np.empty(n_max)
        angles[:] = np.nan
        for j,ow in enumerate(my_waters):

            dist = biased_ion.position - ow.position

            # if the water is on the other side of the box, move it back
            for d in range(3):
                v = np.array([0,0,0])
                v[d] = 1
                if dist[d] >= self.universe.dimensions[d]/2:
                    ow.residue.atoms.translate(v*self.universe.dimensions[d])
                elif dist[d] <= -self.universe.dimensions[d]/2:
                    ow.residue.atoms.translate(-v*self.universe.dimensions[d])

            # calculate and save angles
            pos = ow.position
            bonded_Hs = ow.bonded_atoms
            tmp_pt = bonded_Hs.positions.mean(axis=0)

            v1 = biased_ion.position - pos
            v2 = pos - tmp_pt
            angles[j] = get_angle(v1, v2)*180/np.pi

        return angles


    def _polyhedron_size_per_frame(self, frame_idx, biased_ion, r0=3.15, for_visualization=False):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        biased_ion : str, MDAnalysis.Atom
            Biased ion in the simulation to calculate polyhedrons for
        r0 : float
            Hydration shell cutoff for the biased ion in Angstroms, default=3.15
        for_visualization : bool
            Whether to use this function to output points for visualization, default=False

        Returns
        -------
        area, volume : float
            Volume and maximum cross-sectional area for the polyhedron
        
        '''
    
        # make biased_ion into MDAnalysis Atom
        if isinstance(biased_ion, str):
            ion = self.universe.select_atoms(biased_ion)[0]
        else:
            ion = biased_ion
        
        # initialize the frame
        self.universe.trajectory[frame_idx]

        # Unwrap the shell and generate points on atomic spheres
        shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
        pos = self._unwrap_shell(ion, r0)
        shell.positions = pos
        pos = self._points_on_atomic_radius(shell, n_points=200)
        center = ion.position

        # Create the polyhedron with a ConvexHull and save volume
        hull = ConvexHull(pos)
        volume = hull.volume

        # Get the major axis (first principal component)
        pca = PCA(n_components=3).fit(pos[hull.vertices])

        # Find all the edges of the convex hull
        edges = []
        for simplex in hull.simplices:
            for s in range(len(simplex)):
                edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                edges.append(edge)

        # Create a line through the polyhedron along the principal component
        d = distances.distance_array(shell, shell, box=self.universe.dimensions)
        t_values = np.linspace(-d.max()/2, d.max()/2, 100)
        center_line = np.array([center + t*pca.components_[0,:] for t in t_values])

        # Find the maximum cross-sectional area along the line through polyhedron
        area = 0
        for pt in center_line:

            # Find the plane normal to the principal component
            A, B, C, D = create_plane_from_point_and_normal(pt, pca.components_[0,:])

            # Find the intersection points of the hull edges with the slicing plane
            intersection_points = []
            for edge in edges:
                p1 = pos[edge[0]]
                p2 = pos[edge[1]]
                intersection_point = line_plane_intersection(p1, p2, A, B, C, D)
                if intersection_point is not None:
                    intersection_points.append(intersection_point)

            # If a slicing plane exists and its area is larger than any other, save
            if len(intersection_points) > 0:
                intersection_points = np.array(intersection_points)
                projected_points, rot_mat, mean_point = project_to_plane(intersection_points)
                intersection_hull = ConvexHull(projected_points)

                if intersection_hull.volume > area:
                    saved_points = (pt, intersection_points, projected_points, mean_point)
                    area = intersection_hull.volume

        if for_visualization:
            return area, volume, saved_points
        else:
            return area, volume
    

    def _unwrap_shell(self, ion, r0):
        '''
        Unwrap the hydration shell, so all coordinated groups are on the same side of the box as ion.

        Parameters
        ----------
        ion : MDAnalysis.Atom
            Ion whose shell to unwrap
        r0 : float
            Hydration shell radius for the ion

        Returns
        -------
        positions : np.ndarray
            Unwrapped coordinated for the atoms in the shell

        '''

        dims = self.universe.dimensions
        shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
        dist = ion.position - shell.positions

        correction = np.where(np.abs(dist) > dims[:3]/2, # check if beyond half-box distance
                              np.sign(dist) * dims[:3],  # add or subtract box size based on sign of dist
                              0)                         # otherwise, do not move

        return shell.positions + correction


    def _points_on_atomic_radius(self, shell, n_points=200):
        '''
        Generate points on the "surface" of the atoms, so we have points that encompass the volume of the atoms.

        Parameters
        ----------
        shell : MDAnalysis.AtomGroup
            Hydration shell with all atoms to generate points

        Returns
        -------
        positions : np.ndarray
            Points on the "surface" of the atoms
        '''

        # get all radii
        radii = np.array([self.vdW_radii[atom.element] for atom in shell])
            
        # randomly sample theta and phi angles
        rng = np.random.default_rng()
        theta = np.arccos(rng.uniform(-1,1, (len(shell),n_points)))
        phi = rng.uniform(0,2*np.pi, (len(shell),n_points))

        # convert to Cartesian coordinates
        x = radii[:,None] * np.sin(theta) * np.cos(phi) + shell.positions[:,0,None]
        y = radii[:,None] * np.sin(theta) * np.sin(phi) + shell.positions[:,1,None]
        z = radii[:,None] * np.cos(theta) + shell.positions[:,2,None]

        positions = np.stack((x,y,z), axis=-1)
        return positions.reshape(-1,3)


    def _get_spline_minima(self):
        '''
        Get the spline minima by solving df/dx = 0. Root-finding only works for cubic splines
        for FITPACK (the backend software), so the spline must be a 4th degree spline

        Returns
        -------
        minima_locs : np.array
            Locations of the minima
        minima_vals : np.array
            Values of the minima

        '''

        minima_locs = self.spline.derivative(1).roots()
        minima_vals = self.spline(minima_locs)

        return minima_locs, minima_vals