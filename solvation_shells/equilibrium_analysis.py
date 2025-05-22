# EquilibriumAnalysis class to perform analysis on unbiased MD

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from MDAnalysis.analysis.base import Results

import multiprocessing
from multiprocessing import Pool
from functools import partial

from scipy.spatial import ConvexHull
from scipy.signal import find_peaks
from sklearn.decomposition import PCA

from solvation_analysis.solute import Solute

from utils.linear_algebra import *
from utils.file_rw import vdW_radii
from utils.ParallelMDAnalysis import ParallelInterRDF as InterRDF

class EquilibriumAnalysis:

    def __init__(self, top, traj, water='type OW', cation='resname NA', anion='resname CL'):
        '''
        Initialize the equilibrium analysis object with a topology and a trajectory from
        a production simulation with standard MD
        
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
            
        '''

        self.universe = mda.Universe(top, traj)
        self.n_frames = len(self.universe.trajectory)
        self.waters = self.universe.select_atoms(water)
        self.cations = self.universe.select_atoms(cation)
        self.anions = self.universe.select_atoms(anion)

        if len(self.waters) == 0:
            raise ValueError(f'No waters found with selection {water}')
        if len(self.cations) == 0:
            raise ValueError(f'No cations found with selection {cation}')
        if len(self.anions) == 0:
            raise ValueError(f'No anions found with selection {anion}')

        self.vdW_radii = vdW_radii().get_dict()

        
    def __repr__(self):
        return f'EquilibriumAnalysis object with {len(self.waters)} waters, {len(self.cations)} cations, and {len(self.anions)} anions over {self.n_frames} frames'
    

    def _find_peaks_wrapper(self, bins, data, **kwargs):
        '''Wrapper for scipy.signal.find_peaks to use with SolvationAnalysis to find cutoff'''
        
        peaks, _  = find_peaks(-data, **kwargs)
        radii = bins[peaks[0]]
        return radii
    

    def initialize_Solutes(self, step=1, **kwargs):
        '''
        Initialize the Solute objects from SolvationAnalysis for the ions. Saves the solutes
        in attributes `solute_ci` (cation) and `solute_ai` (anion). 
        
        Parameters
        ----------
        step : int
            Trajectory step for which to run the Solute
        
        '''
        
        self.solute_ci = Solute.from_atoms(self.cations, {'water' : self.waters, 'coion' : self.anions}, 
                                           solute_name='Cation', rdf_kernel=self._find_peaks_wrapper, 
                                           kernel_kwargs={'distance':5}, **kwargs)
        self.solute_ai = Solute.from_atoms(self.anions, {'water' : self.waters, 'coion' : self.cations}, 
                                  solute_name='Anion', rdf_kernel=self._find_peaks_wrapper, 
                                  kernel_kwargs={'distance':5}, **kwargs)

        self.solute_ci.run(step=step)
        self.solute_ai.run(step=step)

        print(f"\nHydration shell cutoff for cation-water = {self.solute_ci.radii['water']:.6f}")
        print(f"Hydration shell cutoff for anion-water = {self.solute_ai.radii['water']:.6f}")


    def determine_ion_pairing_cutoffs(self, find_peaks_kwargs={'distance' : 5, 'height' : -1.1}, plot=True):
        '''
        Calculate the cation-anion radial distributions using SolvationAnalysis and identify the cutoffs for
        ion pairing events. Should plot to ensure the cutoff regimes visually look correct, since these are 
        sensitive to the peak detection algorithm. 

        Parameters
        ----------
        find_peak_kwargs : dict
            Keyword arguments for `scipy.find_peaks` used to find the first 3 minima in the cation-anion RDF,
            default={'distance' : 5, 'height' : -1.1} worked well for NaCl at 0.6 M with OPC3 water
        plot : bool
            Whether to plot the RDF with the regions shaded, default=True

        '''

        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        r = self.solute_ci.rdf_data['Cation']['coion'][0]
        rdf = self.solute_ci.rdf_data['Cation']['coion'][1]
        mins, min_props = find_peaks(-rdf, **find_peaks_kwargs)

        self.ion_pairs = Results()
        self.ion_pairs['CIP'] = (0,r[mins[0]])
        self.ion_pairs['SIP'] = (r[mins[0]],r[mins[1]])
        self.ion_pairs['DSIP'] = (r[mins[1]],r[mins[2]])
        self.ion_pairs['FI'] = (r[mins[2]],np.inf)

        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(r, rdf, color='k')

            le = 2
            for i,m in enumerate(mins[:3]):
                ax.fill_betweenx(np.linspace(0,10), le, r[m], alpha=0.25)
                ax.text((le+r[m]) / 2, 8, list(self.ion_pairs.keys())[i], ha='center')
                le = r[m]

            ax.fill_betweenx(np.linspace(0,10), le, 10, alpha=0.25)
            ax.text((le+10) / 2, 8, list(self.ion_pairs.keys())[-1])
            ax.set_xlabel('r ($\mathrm{\AA}$)')
            ax.set_ylabel('g(r)')
            ax.set_xlim(2,10)
            ax.set_ylim(0,9)
            fig.savefig('ion_pairing_cutoffs.png')
            plt.show()

        return self.ion_pairs
    

    def generate_rdfs(self, bin_width=0.05, range=(0,20), step=1, filename=None, njobs=1):
        '''
        Calculate radial distributions for the solution. This method calculates the RDFs for cation-water,
        anion-water, water-water, and cation-anion using MDAnalysis InterRDF. It saves the data in a 
        dictionary attribute `rdfs` with keys 'ci-w', 'ai-w', 'w-w', and 'ci-ai'.

        Parameters
        ----------
        bin_width : float
            Width of the bins for the RDFs, default=0.05 Angstroms
        range : array-like
            Range over which to calculate the RDF, default=(0,20)
        step : int
            Trajectory step for which to calculate the RDF, default=1
        filename : str
            Filename to save RDF data, default=None means do not save to file
        njobs : int
            Number of CPUs to run on, default=1

        Returns
        -------
        rdfs : dict
            Dictionary with all the results from InterRDF
        
        '''

        nbins = int((range[1] - range[0]) / bin_width)
        self.rdfs = {}

        print('\nCalculating cation-water RDF...')
        ci_w = InterRDF(self.cations, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_w.run(step=step, njobs=njobs)
        self.rdfs['ci-w'] = ci_w.results

        print('\nCalculating anion-water RDF...')
        ai_w = InterRDF(self.anions, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        ai_w.run(step=step, njobs=njobs)
        self.rdfs['ai-w'] = ai_w.results

        print('\nCalculating water-water RDF...')
        w_w = InterRDF(self.waters, self.waters, nbins=nbins, range=range, norm='rdf', verbose=True)
        w_w.run(step=step, njobs=njobs)
        self.rdfs['w-w'] = w_w.results

        print('\nCalculating cation-anion RDF...')
        ci_ai = InterRDF(self.cations, self.anions, nbins=nbins, range=range, norm='rdf', verbose=True)
        ci_ai.run(step=step, njobs=njobs)
        self.rdfs['ci-ai'] = ci_ai.results

        if filename is not None:
            data = np.vstack([ci_w.results.bins, ci_w.results.rdf, ai_w.results.rdf, w_w.results.rdf, ci_ai.results.rdf]).T
            np.savetxt(filename, data, header='r (Angstroms), cation-water g(r), anion-water g(r), water-water g(r), cation-anion g(r)')

        return self.rdfs
    

    def get_coordination_numbers(self, step=1):
        '''
        Calculate the water coordination number as a function of time for both cations and anions.
        
        Parameters
        ----------
        step : int
            Trajectory step for which to calculate coordination numbers
        
        Returns
        -------
        avg_CN : np.array
            Average coordination number over the trajectory for [cations, anions]
        
        '''
    
        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        # initialize coordination number as a function of time
        self.coordination_numbers = np.zeros((2,len(self.universe.trajectory[::step])))

        for i,ts in enumerate(self.universe.trajectory[::step]):
            # first for cations
            d = distances.distance_array(self.cations, self.waters, box=ts.dimensions)
            n_coordinating = (d <= self.solute_ci.radii['water']).sum()
            self.coordination_numbers[0,i] = n_coordinating / len(self.cations)

            # then for anions
            d = distances.distance_array(self.anions, self.waters, box=ts.dimensions)
            n_coordinating = (d <= self.solute_ai.radii['water']).sum()
            self.coordination_numbers[1,i] = n_coordinating / len(self.anions)

        return self.coordination_numbers.mean(axis=1)
        

    def shell_probabilities(self, plot=False):
        '''
        Calculate the shell probabilities for each ion. Must first initialize the SolvationAnalysis Solutes.
        
        Parameters
        ----------
        plot : bool
            Whether to plot the distributions of shells, default=False
            
        '''

        try:
            self.solute_ci
        except NameError:
            print('Solutes not initialized. Try `initialize_Solutes()` first')

        df1 = self.solute_ci.speciation.speciation_fraction
        shell = []
        for i in range(df1.shape[0]):
            row = df1.iloc[i]
            shell.append(f'{row.coion:.0f}-{row.water:.0f}')

        df1['shell'] = shell
        self.cation_shells = df1

        df2 = self.solute_ai.speciation.speciation_fraction
        shell = []
        for i in range(df2.shape[0]):
            row = df2.iloc[i]
            shell.append(f'{row.coion:.0f}-{row.water:.0f}')

        df2['shell'] = shell
        self.anion_shells = df2
        
        if plot:
            df = df1.merge(df2, on='shell', how='outer')
            df.plot(x='shell', y=['count_x', 'count_y'], kind='bar', legend=False)
            plt.legend(['Cation', 'Anion'])
            plt.ylabel('probability')
            plt.savefig('shell_probabilities.png')
            plt.show()


    def water_dipole_distribution(self, ion='cation', radius=None, step=1):
        '''
        Calculate the distribution of angles between the water dipole and the oxygen-ion vector

        Parameters
        ----------
        ion : str
            Ion to calculate the distribution for. Options are 'cation' and 'anion'. default='cation'
        radius : float
            Hydration shell cutoff in Angstroms to select waters within hydration shell only, default=None 
            means pull from SolvationAnalysis.solute.Solute
        step : int
            Step to iterate the trajectory when running the analysis, default=10

        Returns
        -------
        angles : np.array
            Angles for all waters coordinated with all ions, averaged over the number of frames

        '''

        # parse arguments
        if ion == 'cation':
            ions = self.cations
        elif ion == 'anion':
            ions = self.anions
        else:
            raise NameError("Options for kwarg ion are 'cation' or 'anion'")
        
        if radius is None:
            if ion == 'cation':
                radius = self.solute_ci.radii['water']
            elif ion == 'anion':
                radius = self.solute_ai.radii['water']

        # loop through frames and ions to get angle distributions 
        angles = []
        for i, ts in enumerate(self.universe.trajectory[::step]):
            for ci in ions:
                my_atoms = self.universe.select_atoms(f'sphzone {radius} index {ci.index}') - ci
                my_waters = my_atoms & self.waters # intersection operator to get the OW from my_atoms

                for ow in my_waters:

                    dist = ci.position - ow.position

                    # if the water is on the other side of the box, move it back
                    for d in range(3):
                        v = np.array([0,0,0])
                        v[d] = 1
                        if dist[d] >= ts.dimensions[d]/2:
                            ow.residue.atoms.translate(v*ts.dimensions[d])
                        elif dist[d] <= -ts.dimensions[d]/2:
                            ow.residue.atoms.translate(-v*ts.dimensions[d])

                    # calculate and save angles
                    pos = ow.position
                    bonded_Hs = ow.bonded_atoms
                    tmp_pt = bonded_Hs.positions.mean(axis=0)

                    v1 = ci.position - pos
                    v2 = pos - tmp_pt
                    ang = get_angle(v1, v2)*180/np.pi
                    angles.append(ang)
        
        return np.array(angles)
    

    def polyhedron_size(self, ion='cation', r0=None, njobs=1, step=1):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        ion : str
            Whether to calculate the volumes and areas for the cations or anions, options are `cation` and `anion`
        r0 : float
            Hydration shell cutoff in Angstroms, default=None means will calculate using `self.initialize_Solutes()`
        njobs : int
            How many processors to run the calculation with, default=1. If greater than 1, use multiprocessing to
            distribute the analysis. If -1, use all available processors.
        step : int
            Trajectory step for analysis

        Returns
        -------
        results : MDAnalysis.analysis.base.Results object
            Volume and area time series, saved in `volumes` and `areas` attributes
        
        '''

        if ion == 'cation':
            ions = self.cations
            if r0 is None:
                try:
                    r0 = self.solute_ci.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')

        elif ion == 'anion':
            ions = self.anions
            if r0 is None:
                try:
                    r0 = self.solute_ai.radii['water']
                except NameError:
                    print('Solutes not initialized. Try `initialize_Solutes()` first')

        else:
            raise NameError("Options for kwarg ion are 'cation' or 'anion'")
        
        # Prepare the Results object
        results = Results()
        results.areas = np.zeros((len(ions), len(self.universe.trajectory[::step])))
        results.volumes = np.zeros((len(ions), len(self.universe.trajectory[::step])))

        if njobs == 1: # run on 1 CPU

            for i,ts in tqdm(enumerate(self.universe.trajectory[::step])):
                a,v = self._polyhedron_size_per_frame(i, ions, r0=r0)
                results.areas[:,i] = a
                results.volumes[:,i] = v

        else: # run in parallel
            
            if njobs == -1:
                n = multiprocessing.cpu_count()
            else:
                n = njobs

            run_per_frame = partial(self._polyhedron_size_per_frame,
                                    ions=ions,
                                    r0=r0)
            frame_values = np.arange(self.universe.trajectory.n_frames, step=step)

            with Pool(n) as worker_pool:
                result = worker_pool.map(run_per_frame, frame_values)

            result = np.asarray(result)
            results.areas = result[:,0,:].T
            results.volumes = result[:,1,:].T

        return results
    

    def _polyhedron_size_per_frame(self, frame_idx, ions, r0):
        '''
        Construct a polyhedron from the atoms in a hydration shell and calculate the volume of the polyhedron
        and the maximum cross-sectional area of the polyhedron. The cross-sections are taken along the first 
        principal component of the vertices of the polyhedron.

        Parameters
        ----------
        frame_idx : int
            Index of the frame
        ions : MDAnalysis.AtomGroup
            Ions in the simulation to calculate polyhedrons for
        r0 : float
            Hydration shell cutoff for the biased ion in Angstroms, default=3.15

        Returns
        -------
        area, volume : np.ndarray
            Volumes and maximum cross-sectional areas for the polyhedrons for each ion, shape is len(ions)
        
        '''

        # initialize the frame
        self.universe.trajectory[frame_idx]

        volumes = np.zeros(len(ions))
        areas = np.zeros(len(ions))

        for j,ion in enumerate(ions):

            # Unwrap the shell
            shell = self.universe.select_atoms(f'(sphzone {r0} index {ion.index})')
            pos = self._unwrap_shell(ion, r0)
            shell.positions = pos
            pos = self._points_on_atomic_radius(shell, n_points=200)
            center = ion.position

            # Create the polyhedron with a ConvexHull and save volume
            hull = ConvexHull(pos)
            volumes[j] = hull.volume

            # Get the major axis (first principal component)
            pca = PCA(n_components=3).fit(pos[hull.vertices])

            # Find all the edges of the convex hull
            edges = []
            for simplex in hull.simplices:
                for s in range(len(simplex)):
                    edge = tuple(sorted((simplex[s], simplex[(s + 1) % len(simplex)])))
                    edges.append(edge)

            # Create a line through the polyhedron along the principal component
            d = distances.distance_array(shell, shell, box=ions.universe.dimensions)
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
                        area = intersection_hull.volume
            
            areas[j] = area

        return areas, volumes


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
            Unwrapped coordinates for the shell

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