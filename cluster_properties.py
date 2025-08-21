from __future__ import division
import sys
sys.path.insert(0, '/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/arepo_hdf5_library_old')
import group_particles
import read_hdf5
import numpy as np
import scipy.optimize as op
import scipy.stats as stat

import matplotlib
from pylab import *
from matplotlib.ticker import FormatStrFormatter
matplotlib.use('Agg')
mpl.rc('text', usetex=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle
import os
import h5py

# useful constants
XH = 0.76   # hydrogen mass fraction
gamma = 5/3   # adiabatic index
mp = 1.67e-27   # atomic mass unit in kg
me = 9.109e-31   # electron mass in kg
joules_to_ergs = 1e7
eV_to_joules = 1.6022e-19
kB = 1.3806e-23   # Boltzmann constant units J/K
kpc_to_m = 3.086e19
year_to_s = 3.156e7   # 365.25 * 24 * 3600
Msun = 1.988e30   # units kg
c = 2.998e8   # m/s
sigma_t = 6.6525e-29   # Thomson cross section for electron in m^2


def gas_temperature(u,xe):
    energy_per_mass_unit_conversion = 1e10   # u CGS units (km/s)^2 -> (cm/s)^2
    mu = (4 / (1 + (3 * XH) + (4 * XH * xe))) * (mp * 1000)   # units grams
    T = (gamma-1) * (u/(kB*joules_to_ergs)) * energy_per_mass_unit_conversion * mu   # Kelvin 
    return T * (kB/eV_to_joules) / 1000   # units keV


def f_R(f_R0, a, omega_m, omega_l):
    '''Returns the background scalaron field value for the Hu-Sawicki f(R) model as a funstion of the scale factor with n=1'''
    c_ratio = (6 * omega_l) / omega_m
    R_m2_ratio = 3 * (a ** (-3) + ((2/3) * c_ratio))
    c_c2_ratio = -f_R0 * np.power(3 * (1 + (4 * omega_l) / omega_m), 2)
    return -c_c2_ratio * np.power(R_m2_ratio, -2)


class FindData:
    def __init__(self, simulation, model, realisation, snapshot, system='cosma7'):
        datapath = self.get_data_path(simulation, model, realisation)
        self.group_path="%sgroups_%03d/fof_subhalo_tab_%03d." % (datapath, snapshot, snapshot)
        self.particle_path=datapath
        self.unit_factor = self.get_unit_factor(simulation)
        print("")
        print("Group directory is " + self.group_path)
        print("Particle directory is " + self.particle_path)
        print("")

    def get_data_path(self, simulation, model, realisation):
        """Locate the file path to simulation data

        :type simulation: str
        :param simulation: name of simulation

        :returns: string with the simulation filepath
        """
        if simulation == "L62_N512":
            return "/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/%s_%s_kpc/" % (simulation, model)
        elif simulation in ["L100_N256","L54_N256","L68_N256","L86_N256","L136_N512"]:
            if system=="cosma7":
                return "/cosma7/data/dp004/dc-mitc2/Cluster_MG_tests/full_physics/%s_%s/%s/" % (simulation, model, realisation)
            elif system=="madfs":
                return "/madfs/data/dc-mitc2/Cluster_MG_tests/full_physics/%s_%s/%s/" % (simulation, model, realisation)
            elif system=="cosma6":
                return "/cosma6/data/dp004/dc-mitc2/Cluster_MG_tests/full_physics/%s_%s/%s/" % (simulation, model, realisation)
        elif simulation == "L302_N1136":
            return "/cosma8/data/dp203/bl267/Data/ClusterSims/%s_%s/" % (simulation, model)
        elif simulation == "L62_N512_non_rad":
            return "/cosma6/data/dp004/dc-arno1/SZ_project/non_radiative_hydro/L62_N512_%s/" % (model)
        if simulation == "L25_N512":
            return "/cosma7/data/dp004/dc-arno1/Shibone/L25N512/full_physics/L25N512_%s/" % (model)
        elif simulation in ["TNG300-3","TNG300-1"]:
            return "/cosma7/data/TNG/%s/" % (simulation)
        elif simulation == "L62_N512_nDGP":
            if model == "GR":
                return "/madfs/data/dc-hern1/L62_N512_Hydro/GR-full/"
            else:
                return "/madfs/data/dc-hern1/L62_N512_Hydro/%s-full_no-ref/" % (model)

    def get_unit_factor(self, simulation):
        """Obtain the distance unit factor

        :type simulation: str
        :param simulation: name of simulation

        :returns: 1000 if simulation uses Mpc distances;
          1 for kpc distances
        """
        if simulation == "L62_N512_non_rad":
            return 1000   # Mpc distances
        else:
            return 1   # kpc distances


class ClusterProperties:
#   def __init__(self, simulation, model, realisation, snapshot, system='cosma7', mass_cut=-1, delta=500, file_ending="highMass_paper", rescaling="true", core_frac=0.15):
    def __init__(self, simulation, model, realisation, snapshot, system='cosma7', mass_cut=1.0e13, delta=500, file_ending="highMass_paper", rescaling="true", core_frac=0.15):
        self.simulation = simulation
        self.model = model
        self.snapshot = snapshot
        self.fileroot = "/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/%s/%s/" % (self.simulation,self.model)
        self.realisation = realisation
        self.mass_cut = mass_cut
        self.delta = delta
        self.file_ending = file_ending
        self.rescaling = rescaling
        self.core_frac = core_frac

        if self.simulation == "L62_N512_non_rad":
            # Simulation is non-radiative
            self.non_rad = True
            self.fp_ptypes = ['gas', 'dm']
            self.fp_parttypes = [0, 1]
        else:
            # Simulation is full-physics
            self.non_rad = False
            self.fp_ptypes = ['gas', 'dm', 'stars', 'bh']
            self.fp_parttypes = [0, 1, 4, 5]
    
        # set unique filename label if core != 0.15
        if self.core_frac == 0.15:
            self.core_label = ""
        else:
            self.core_label = "_%.2f" % (self.core_frac)

        # is the data GR or f(R)?
        if self.model == "GR":
            self.gr = True
        else:
            self.gr = False

        fd = FindData(simulation,model,realisation,snapshot,system=system)
        self.group_path = fd.group_path
        self.particle_path = fd.particle_path
        self.unit_factor = fd.unit_factor

        print("loading data...")
        print("")
        self.load_simulation_data()


    def load_simulation_data(self):
        """Load simulation particle and group data
        """
        # create instance of the snapshot class
        self.s = read_hdf5.snapshot(self.snapshot, directory = self.particle_path, dirbases = ["snapdir_", ""], snapbases = ["/GR_", "/gadget", "/snap_"], check_total_particle_number = True, verbose = False)

        print("hdf reading finished")

        # load group and particle data
        if self.non_rad:
            self.s.read(['Coordinates', 'Velocities', 'Masses', 'Density', 'ElectronAbundance', 'InternalEnergy'], parttype = -1)
        else:
            self.s.read(['Coordinates', 'Velocities', 'Masses', 'Density', 'ElectronAbundance', 'InternalEnergy', 'StarFormationRate', 'GFM_StellarFormationTime', 'GFM_InitialMass', 'CoolingHeatingEnergy'], parttype = self.fp_parttypes)
        if self.model[0] == "F" and self.snapshot == 50:
            self.s.group_catalog(["GroupPos", "Group_M_Crit200", "Group_M_Crit500", "Group_R_Crit200", "Group_R_Crit500", "GroupNsubs", "GroupLenType", "Group_M_Eff_In_R_Crit500", "Group_M_Vir_Eff", "Group_R_Vir_Eff", "Group_M_In_R_Vir_Eff", "SubhaloPos", "SubhaloLenType", "SubhaloMassInRadType", "SubhaloMassType"], path = self.group_path)
        else:
            self.s.group_catalog(["GroupPos", "Group_M_Crit200", "Group_M_Crit500", "Group_R_Crit200", "Group_R_Crit500", "GroupNsubs", "GroupLenType", "SubhaloPos", "SubhaloLenType", "SubhaloMassInRadType", "SubhaloMassType"], path = self.group_path)

        # Cosmological scale factor: all quantities below have proper units
        scale_factor = 1 / (1 + self.s.header.redshift)

        # FoF group data
        self.group_pos = self.s.cat['GroupPos'] * scale_factor * self.unit_factor  # kpc
        self.group_r200 = self.s.cat['Group_R_Crit200'] * scale_factor * self.unit_factor  # kpc
        self.group_m200 = self.s.cat['Group_M_Crit200']   # Msun
        if self.model[0] == "F" and self.snapshot == 45:
            # L62 f(R) data includes effective mass for z=0
            if self.rescaling == "effective":
                print("Using effective rescaling")
                # Use masses computed within R500,eff
                self.group_r500 = self.s.cat['Group_R_Vir_Eff'] * scale_factor * self.unit_factor  # kpc
                self.group_m500 = self.s.cat['Group_M_In_R_Vir_Eff']  # Msun
                self.group_meff500 = self.s.cat['Group_M_Vir_Eff']  # Msun
            else:
                print("Using true rescaling")
                # Use masses computed within R500,true
                self.group_r500 = self.s.cat['Group_R_Crit500'] * scale_factor * self.unit_factor  # kpc
                self.group_m500 = self.s.cat['Group_M_Crit500']  # Msun
                self.group_meff500 = self.s.cat['Group_M_Eff_In_R_Crit500']  # Msun
        else:
            #If not f(R) gravity, no effective mass or effective field
            self.group_r500 = self.s.cat['Group_R_Crit500'] * scale_factor * self.unit_factor  # kpc
            self.group_m500 = self.s.cat['Group_M_Crit500']  # Msun

        self.group_nsubs = self.s.cat['GroupNsubs']
        self.group_nstars = self.s.cat['GroupLenType'][:, 4]

        # subhalo data
        self.sub_nstars = self.s.cat['SubhaloLenType'][:, 4]
        self.sub_pos = self.s.cat['SubhaloPos'] * scale_factor * self.unit_factor  # kpc
        print("Max (proper) sub coordinate [kpc]: ", max(self.sub_pos[:, 0]), max(self.sub_pos[:, 1]), max(self.sub_pos[:, 2]))

        # particle data
        self.gas_positions = self.s.data['Coordinates']['gas'] * scale_factor * self.unit_factor  # kpc
        self.dm_positions = self.s.data['Coordinates']['dm'] * scale_factor * self.unit_factor  # kpc
        self.gas_masses = self.s.data['Masses']['gas']  # Msun
        self.gas_densities = self.s.data['Density']['gas'] * (self.s.header.hubble ** 2 * 1e10) / ((scale_factor * self.unit_factor) ** 3)  # units Msun/(kpc^3)
        self.dm_masses = self.s.data['Masses']['dm']   # Msun
        self.internal_energy = self.s.data['InternalEnergy']['gas']

        # full-ph simulation includes electron abundance of gas
        # also includes stars and black holes
        if not self.non_rad:
            print("Radiative simulation: exact e abundance is used.")
            self.gas_sfr = self.s.data['StarFormationRate']['gas']  # Msun yr^-1
            self.electron_abundance = self.s.data['ElectronAbundance']['gas']
            if 'stars' in self.fp_ptypes:
                self.star_positions = self.s.data['Coordinates']['stars'] * scale_factor * self.unit_factor  # kpc
                self.star_gfmSFT = self.s.data['GFM_StellarFormationTime']['stars']
                self.star_masses = self.s.data['Masses']['stars']  # Msun
                self.star_ini_masses = self.s.data['GFM_InitialMass']['stars'] * 1e10 / self.s.header.hubble  # Msun
                print("Median star mass: ", np.median(self.star_masses))
                print("Mean star mass: ", np.mean(self.star_masses))
            if 'bh' in self.fp_ptypes:
                self.bh_positions = self.s.data['Coordinates']['bh'] * scale_factor * self.unit_factor  # kpc
                self.bh_masses = self.s.data['Masses']['bh']  # Msun

        else:
            self.electron_abundance = np.full(len(self.internal_energy), 1.158)
            # 1.158 assumes primordial H fraction and completely ionised gas
            # assumes two electrons from each He and one from each H
            print("Non-radiative simulation: fix e abundance at 1.158 per H.")


        if self.delta == "all":
            # all group masses included
            group_part_file = "%sgroup_particles/all_groups_%s_" % (self.fileroot, self.realisation)
        else:
            group_part_file = "%sgroup_particles/%s_%s_" % (self.fileroot, self.file_ending, self.realisation)

        # load grouped particles
        if self.non_rad:
            self.gp = group_particles.GroupParticles(self.s, group_part_file, self.snapshot, ptypes = ['gas', 'dm'], parttypes = [0, 1], mass_cut = self.mass_cut, delta = self.delta)
        else:
            self.gp = group_particles.GroupParticles(self.s, group_part_file, self.snapshot, ptypes=self.fp_ptypes, parttypes=self.fp_parttypes, mass_cut=self.mass_cut, delta=self.delta)


    def calc_electron_pressure(self, group_id = -1, Lbox = 301.75, Ngrid = 256):
        if self.model == "GR" or self.simulation == "L302_N1136":
            pressure_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s%s_pressure.pickle" % (self.simulation, self.model, self.realisation, self.snapshot, self.file_ending, self.core_label)
            print(pressure_dumpfile)
        else:
            pressure_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s_rescaling%s%s_pressure.pickle" % (self.simulation, self.model, self.realisation, self.snapshot, self.file_ending, self.rescaling, self.core_label)

        # define logarithmic radial bins, units kpc
        self.nbins = 28
        min_rad = 14.9
        max_rad = 1000
        self.bins = np.logspace(np.log10(min_rad), np.log10(max_rad), self.nbins + 1, base = 10.0)
        self.bin_radii = 0.5 * (self.bins[1:] + self.bins[:-1])   
        # midpoint of each bin, kpc units

        # define group sample
        if group_id != -1:
            self.sample = [group_id]
        else:
            if self.delta == 200:
                self.sample = np.where(self.s.cat['Group_M_Crit200'] * self.s.header.hubble > self.mass_cut)[0]
            elif self.delta == 500:
                self.sample = np.where(self.s.cat['Group_M_Crit500'] * self.s.header.hubble > self.mass_cut)[0]
            elif self.delta == "all":
                self.sample = np.where(self.s.cat['Group_M_Crit500'] * self.s.header.hubble > -inf)[0]
        # NB:// mass_cut units are Msun/h

        print("Median M500 [Msun/h]: %.2f" % (np.log10(np.median(self.group_m500[self.sample] * self.s.header.hubble))))
        print("Min / Max mass: %.2f / %.2f" % (np.log10(min(self.group_m500[self.sample] * self.s.header.hubble)), np.log10(max(self.group_m500[self.sample] * self.s.header.hubble))))
        print("Sample size: ", len(self.sample))

        print("Processing group properties ...")
        cell_counter = 0
        sfr_counter = 0
        positions_all = np.empty((0, self.gas_positions.shape[1]))
        electron_pressure_total = np.array([])
        for (index, group) in enumerate(self.sample):
            if index % 100 == 0:
                print("Iteration %s/%s" % (index, len(self.sample)))
               
            gas_radii = np.zeros(len(self.gp.group_particles['gas'][group]))
            temp = np.zeros(len(self.gp.group_particles['gas'][group]))
            electron_number = np.zeros(len(self.gp.group_particles['gas'][group]))
            positions = np.zeros((len(self.gp.group_particles['gas'][group]), self.gas_positions.shape[1]))
            electron_pressure = np.zeros(len(self.gp.group_particles['gas'][group]))

            # iterate through gas particles within r<R200 in each group
            for (pid, particle) in enumerate(self.gp.group_particles['gas'][group]):
                cell_counter += 1
                ### For referee ###
                if self.gas_sfr[particle] > 0.:
                    sfr_counter += 1
#                    continue   # Skip particle if SFR > 0 (i.e., temp will remain zero)
                ### End ###

                gas_radii[pid] = np.sqrt(np.sum((self.gas_positions[particle] - self.group_pos[group])**2))
                temp[pid] = gas_temperature(self.internal_energy[particle], self.electron_abundance[particle])   # units keV
                electron_number[pid] = self.electron_abundance[particle] * (XH * self.gas_masses[particle] / (mp / Msun))
                # right-hand bracket gives number of hydrogen atoms
                positions[pid] = self.gas_positions[particle]
                electron_pressure[pid] = temp[pid] * electron_number[pid] / (self.gas_masses[particle] / self.gas_densities[particle])   # units keV kpc^-3
            positions_all = np.vstack([positions_all, positions])
            electron_pressure_total = np.concatenate([electron_pressure_total, electron_pressure])
        print(f"Electron pressure total: {electron_pressure_total.shape}")
        print(f"Position all: {positions_all.shape}")

        # Grid electron pressure
        x = positions_all[:,0]
        y = positions_all[:,1]
        z = positions_all[:,2]

        x = np.mod(x,Lbox)  # Wrap positions into [0, Lbox)
        y = np.mod(y,Lbox)
        z = np.mod(z,Lbox)

        edges = [np.linspace(0, Lbox, Ngrid+1)] * 3  # Bin edges for the grid

        Pe_grid_sum, _ = np.histogramdd(  # Sum pressure into voxels
            sample=np.vstack([x, y, z]).T,
            bins=edges,
            weights=electron_pressure_total
        )
        print(f"Pressure grid: {Pe_grid_sum}, shape: {Pe_grid_sum.shape}")

        # Save data
        df = open(pressure_dumpfile,"wb+")
        pickle.dump((Pe_grid_sum, Lbox, Ngrid),df)
        df.close
        print(f"Saved at: {pressure_dumpfile}")


    def cluster_properties(self, group_id = -1):
        if self.model == "GR" or self.simulation == "L302_N1136":
            group_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s%s.pickle" % (self.simulation, self.model, self.realisation, self.snapshot, self.file_ending, self.core_label)
            subhalo_dumpfile = self.fileroot+"pickle_files/subhalo_%s_%s_%s_s%d_%s.pickle" % (self.simulation,self.model,self.realisation,self.snapshot,self.file_ending)
            print(group_dumpfile)
            print(subhalo_dumpfile)
        else:
            group_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s_rescaling%s%s.pickle" % (self.simulation, self.model, self.realisation, self.snapshot, self.file_ending, self.rescaling, self.core_label)
            subhalo_dumpfile = self.fileroot+"pickle_files/subhalo_%s_%s_%s_s%d_%s_rescaling%s.pickle" % (self.simulation,self.model,self.realisation,self.snapshot,self.file_ending,self.rescaling)

        # define logarithmic radial bins, units kpc
        self.nbins = 28
        min_rad = 14.9
        max_rad = 1000
        self.bins = np.logspace(np.log10(min_rad), np.log10(max_rad), self.nbins + 1, base = 10.0)
        self.bin_radii = 0.5 * (self.bins[1:] + self.bins[:-1])   
        # midpoint of each bin, kpc units

        # define group sample
        if group_id != -1:
            self.sample = [group_id]
        else:
            if self.delta == 200:
                self.sample = np.where(self.s.cat['Group_M_Crit200'] * self.s.header.hubble > self.mass_cut)[0]
            elif self.delta == 500:
                self.sample = np.where(self.s.cat['Group_M_Crit500'] * self.s.header.hubble > self.mass_cut)[0]
            elif self.delta == "all":
                self.sample = np.where(self.s.cat['Group_M_Crit500'] * self.s.header.hubble > -inf)[0]
        # NB:// mass_cut units are Msun/h

        print("Median M500 [Msun/h]: %.2f" % (np.log10(np.median(self.group_m500[self.sample] * self.s.header.hubble))))
        print("Min / Max mass: %.2f / %.2f" % (np.log10(min(self.group_m500[self.sample] * self.s.header.hubble)), np.log10(max(self.group_m500[self.sample] * self.s.header.hubble))))
        print("Sample size: ", len(self.sample))

        if not os.path.exists(group_dumpfile):
            print("Dumpfile %s does not exist" % (group_dumpfile))

            # initialise property arrays
            self.vol_temp_profile = np.zeros((len(self.sample),self.nbins))
            self.mass_temp_profile = np.zeros((len(self.sample),self.nbins))
            self.density_profile = np.zeros((len(self.sample),self.nbins))
            self.electron_pressure_profile = np.zeros((len(self.sample),self.nbins))
            self.cum_fgas = np.zeros((len(self.sample),self.nbins+1))
            self.vol_T500 = np.zeros(len(self.sample))
            self.mass_T500 = np.zeros(len(self.sample))
            self.vol_T500_with_core = np.zeros(len(self.sample))
            self.mass_T500_with_core = np.zeros(len(self.sample))
            self.Mg500 = np.zeros(len(self.sample))
            self.Mstar = np.zeros(len(self.sample))
            self.A19_Mstar = np.zeros(len(self.sample))
            self.SMF =  np.zeros(len(self.sample))
            self.Ysz_with_core = np.zeros(len(self.sample))
            self.Ysz_no_core = np.zeros(len(self.sample))
            self.Lx_with_core = np.zeros(len(self.sample))
            self.Lx_no_core = np.zeros(len(self.sample))

            print("Processing group properties ...")
            cell_counter = 0
            sfr_counter = 0
            for (index, group) in enumerate(self.sample):
                if index % 100 == 0:
                    print("Iteration %s/%s" % (index, len(self.sample)))
               
                gas_radii = np.zeros(len(self.gp.group_particles['gas'][group]))
                temp = np.zeros(len(self.gp.group_particles['gas'][group]))
                electron_number = np.zeros(len(self.gp.group_particles['gas'][group]))
                electron_pressure = np.zeros(len(self.gp.group_particles['gas'][group]))

                # iterate through gas particles within r<R200 in each group
                for (pid, particle) in enumerate(self.gp.group_particles['gas'][group]):
                    cell_counter += 1
                    ### For referee ###
                    if self.gas_sfr[particle] > 0.:
                        sfr_counter += 1
#                        continue   # Skip particle if SFR > 0 (i.e., temp will remain zero)
                    ### End ###

                    gas_radii[pid] = np.sqrt(np.sum((self.gas_positions[particle] - self.group_pos[group])**2))
                    temp[pid] = gas_temperature(self.internal_energy[particle], self.electron_abundance[particle])   # units keV
                    electron_number[pid] = self.electron_abundance[particle] * (XH * self.gas_masses[particle] / (mp / Msun))
                    # right-hand bracket gives number of hydrogen atoms
                    electron_pressure[pid] = temp[pid] * electron_number[pid] / (self.gas_masses[particle] / self.gas_densities[particle])   # units keV kpc^-3

                gas_mass = self.gas_masses[self.gp.group_particles['gas'][group]]
                gas_density = self.gas_densities[self.gp.group_particles['gas'][group]]

                dm_radii = np.zeros(len(self.gp.group_particles['dm'][group]))
                for (pid, particle) in enumerate(self.gp.group_particles['dm'][group]):
                    dm_radii[pid] = np.sqrt(np.sum((self.dm_positions[particle] - self.group_pos[group])**2))

                dm_mass = self.dm_masses[self.gp.group_particles['dm'][group]]

                # reorder gas data by radius
                order_gas_indices = np.argsort(gas_radii)
                ordered_gas_radii = gas_radii[order_gas_indices]
                ordered_gas_temp = temp[order_gas_indices]
                ordered_gas_mass = gas_mass[order_gas_indices]    
                ordered_gas_density = gas_density[order_gas_indices]
                ordered_electron_number = electron_number[order_gas_indices]
                ordered_electron_pressure = electron_pressure[order_gas_indices]

                # reorder DM data by radius
                order_dm_indices = np.argsort(dm_radii)
                ordered_dm_radii = dm_radii[order_dm_indices]
                ordered_dm_mass = dm_mass[order_dm_indices]

                # gas and DM counts in each radius bin
                npart = np.histogram(ordered_gas_radii, self.bins)[0]
                dm_npart = np.histogram(ordered_dm_radii, self.bins)[0]
                
                # initialise counts and masses using particles within min radius
                count = len(ordered_gas_mass[ordered_gas_radii < self.bins[0]])
                dm_count = len(ordered_dm_mass[ordered_dm_radii < self.bins[0]])
                cum_mgas = np.sum(ordered_gas_mass[ordered_gas_radii < self.bins[0]])
                cum_mdm = np.sum(ordered_dm_mass[ordered_dm_radii < self.bins[0]])
                self.cum_fgas[index][0] = cum_mgas / (cum_mgas + cum_mdm)

                # iterate over radial bins
                for i in range(self.nbins):
                    # volume-weighted temperature of each bin
                    self.vol_temp_profile[index][i] = np.sum(ordered_gas_temp[count : count + npart[i]] * (ordered_gas_mass[count : count + npart[i]] / ordered_gas_density[count : count + npart[i]])) / np.sum(ordered_gas_mass[count : count + npart[i]] / ordered_gas_density[count : count + npart[i]])

                    # mass-weighted temperature of each bin
                    self.mass_temp_profile[index][i] = np.sum(ordered_gas_temp[count : count + npart[i]] * ordered_gas_mass[count : count + npart[i]]) / np.sum(ordered_gas_mass[count : count + npart[i]])

                    # average density of each bin
                    self.density_profile[index][i] = np.sum(ordered_gas_mass[count : count + npart[i]]) / ((4/3) * np.pi * (self.bins[i+1]**3 - self.bins[i]**3) / 1000**3)
                    # density units Msun / Mpc^3

                    # volume-weighted electron pressure of each bin
                    self.electron_pressure_profile[index][i] = np.sum(ordered_electron_pressure[count : count + npart[i]] * (ordered_gas_mass[count : count + npart[i]] / ordered_gas_density[count : count + npart[i]])) / np.sum(ordered_gas_mass[count : count + npart[i]] / ordered_gas_density[count : count + npart[i]])

                    cum_mgas += np.sum(ordered_gas_mass[count : count + npart[i]])
                    cum_mdm += np.sum(ordered_dm_mass[dm_count : dm_count + dm_npart[i]])
                    # total gas and DM masses enclosed by bin edge i+1
                    self.cum_fgas[index][i+1] = cum_mgas / (cum_mgas + cum_mdm)
                    count += npart[i]
                    dm_count += dm_npart[i]


                # gas cell indices between core_frac * R500 and R500
                R500_no_core_gas_sample = np.where((self.core_frac * self.group_r500[group] <= ordered_gas_radii) & (ordered_gas_radii <= self.group_r500[group]))

                # volume-weighted temperature
                self.vol_T500[index] = np.sum(ordered_gas_temp[R500_no_core_gas_sample] * (ordered_gas_mass[R500_no_core_gas_sample] / ordered_gas_density[R500_no_core_gas_sample])) / np.sum(ordered_gas_mass[R500_no_core_gas_sample] / ordered_gas_density[R500_no_core_gas_sample])

                # mass-weighted temperature
                self.mass_T500[index] = np.sum(ordered_gas_temp[R500_no_core_gas_sample] * ordered_gas_mass[R500_no_core_gas_sample]) / np.sum(ordered_gas_mass[R500_no_core_gas_sample])

                # Ysz parameter
                self.Ysz_no_core[index] = ((sigma_t/(kpc_to_m*1000)**2)/(me*c**2/(eV_to_joules*1000)))*np.sum(ordered_electron_number[R500_no_core_gas_sample]*ordered_gas_temp[R500_no_core_gas_sample])   # units Mpc^2
                # mass-weighted temperature (keV) of gas enclosed within 0.15-1.R500

                # Lx
                self.Lx_no_core[index] = np.sum(ordered_gas_mass[R500_no_core_gas_sample] * (ordered_gas_density[R500_no_core_gas_sample] * 1000**3) * (ordered_gas_temp[R500_no_core_gas_sample]**0.5))   # units keV^1/2 Msun^2 Mpc^-3
               
 
                # all gas cell indices within R500 (inc. core)
                R500_gas_sample = np.where(ordered_gas_radii<=self.group_r500[group])

                # total gas mass (Msun) enclosed within R500
                self.Mg500[index] = np.sum(ordered_gas_mass[R500_gas_sample])

                # volume-weighted temperature (inc. core)
                self.vol_T500_with_core[index] = np.sum(ordered_gas_temp[R500_gas_sample] * (ordered_gas_mass[R500_gas_sample] / ordered_gas_density[R500_gas_sample])) / np.sum(ordered_gas_mass[R500_gas_sample] / ordered_gas_density[R500_gas_sample])

                # mass-weighted temperature (inc. core)
                self.mass_T500_with_core[index] = np.sum(ordered_gas_temp[R500_gas_sample] * ordered_gas_mass[R500_gas_sample]) / np.sum(ordered_gas_mass[R500_gas_sample])

                # Ysz parameter (inc. core)
                self.Ysz_with_core[index] = ((sigma_t/(kpc_to_m*1000)**2)/(me*c**2/(eV_to_joules*1000)))*np.sum(ordered_electron_number[R500_gas_sample] * ordered_gas_temp[R500_gas_sample])   # units Mpc^2

                # Lx
                self.Lx_with_core[index] = np.sum(ordered_gas_mass[R500_gas_sample] * (ordered_gas_density[R500_gas_sample] * 1000**3) * (ordered_gas_temp[R500_gas_sample]**0.5))   # units keV^1/2 Msun^2 Mpc^-3

                # if full-ph simulation, find stellar and BH properties
                if not self.non_rad:
                    # find radii of stellar particles
                    star_radii = np.zeros(len(self.gp.group_particles['stars'][group]))
                    for (pid,particle) in enumerate(self.gp.group_particles['stars'][group]):
                        star_radii[pid] = np.sqrt(np.sum((self.star_positions[particle]-self.group_pos[group])**2))   # kpc

                    star_mass = self.star_masses[self.gp.group_particles['stars'][group]]

#                    if 'bh' in fp_ptypes:
                        # find radii of BH particles
#                        bh_radii = np.zeros(len(self.gp.group_particles['bh'][group]))
#                        for (pid,particle) in enumerate(self.gp.group_particles['bh'][group]):
#                            bh_radii[pid] = np.sqrt(np.sum((self.bh_positions[particle]-self.group_pos[group])**2))   # kpc
#                        bh_mass = self.bh_masses[self.gp.group_particles['bh'][group]]
#                        order_bh_indices = np.argsort(bh_radii)
#                        ordered_bh_radii = bh_radii[order_bh_indices]
#                        ordered_bh_mass = bh_mass[order_bh_indices]

                    # order star particles by radius
                    order_star_indices = np.argsort(star_radii)
                    ordered_star_radii = star_radii[order_star_indices]
                    ordered_star_mass = star_mass[order_star_indices]

                    # use star particles within R500
                    R500_star_sample = np.where(ordered_star_radii<=self.group_r500[group])
                    self.Mstar[index] = np.sum(ordered_star_mass[R500_star_sample])

                    # use star particles within 30kpc
                    A19_30kpc_star_sample = np.where(ordered_star_radii<30)
                    A19_stellar_mass = np.sum(ordered_star_mass[A19_30kpc_star_sample])

                    # stellar mass [< 30 kpc] / total mass [M200c]
                    self.SMF[index] = A19_stellar_mass / self.group_m200[group]
                    self.A19_Mstar[index] = A19_stellar_mass   # stellar mass within 30kpc

            print("Star-forming cells: ", sfr_counter)
            print("Total cells: ", len(self.gas_sfr))
            print("Total group cells: ", cell_counter)

#           df = open(group_dumpfile,"w+")
            df = open(group_dumpfile,"wb+")
            if self.non_rad:
                pickle.dump((self.group_m500[self.sample], self.group_m200[self.sample], self.Mg500, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas), df)
            else:
                pickle.dump((self.group_m500[self.sample], self.group_m200[self.sample], self.Mg500, self.Mstar, self.A19_Mstar, self.SMF, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas), df)
            df.close()

        else:
            print("%s exists!" % (group_dumpfile))
#           df = open(group_dumpfile, 'r')
            df = open(group_dumpfile, 'rb')
            if self.non_rad:
                (self.group_m500[self.sample], self.group_m200[self.sample], self.Mg500, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas) = pickle.load(df)
            else:
                (self.group_m500[self.sample], self.group_m200[self.sample], self.Mg500, self.Mstar, self.A19_Mstar, self.SMF, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas) = pickle.load(df)
            df.close()

        if not self.non_rad:
            # only have stars in full-ph simulations
            if not os.path.exists(subhalo_dumpfile):
                print("Subhalo dumpfile %s does not exist" % (subhalo_dumpfile))

                # compute subhalo stellar masses (for SM function)
                print("Computing stellar masses of subhaloes")
                self.sub_starmass = np.zeros(len(self.sub_pos))  # initialise at zero mass for each subhalo
                sub_count = 0   # number of subhaloes in groups studied so far
                tot_star_count = 0   # total number of stars in groups studied so far
                for group in range(len(self.group_pos)):
                    if group % 500 == 0:
                        print("Group %d / %d" % (group, len(self.group_pos)))
                    group_star_count = 0   # number of stars in group sub-haloes studied so far
                    for sub in range(int(self.group_nsubs[group])):
                        sub_id = sub_count + sub
                        stellar_mass = 0   # stellar mass of subhalo
                        for part in range(int(self.sub_nstars[sub_id])):
                            part_id = tot_star_count + group_star_count + part
                            radius = np.sqrt(np.sum((self.star_positions[part_id] - self.sub_pos[sub_id])**2))   # kpc
                            if radius <= 30. * (1 + self.s.header.redshift):   # physics radius 30kpc
                                stellar_mass += self.star_masses[part_id]   # Msun
                        self.sub_starmass[sub_id] = stellar_mass
                        group_star_count += int(self.sub_nstars[sub_id])
                    sub_count += int(self.group_nsubs[group])
                    tot_star_count += int(self.group_nstars[group])

#               df = open(subhalo_dumpfile,"w+")
                df = open(subhalo_dumpfile,"wb+")
                pickle.dump((self.sub_starmass), df)
                df.close()
            else:
                print("Subhalo dumpfile %s exists" % (subhalo_dumpfile))
#               df = open(subhalo_dumpfile,"r")
                df = open(subhalo_dumpfile,"rb")
                (self.sub_starmass) = pickle.load(df)
                df.close()


    def profile(self, mass_bin = "low"):
#       profile_dumpfile = "%s%s/pickle_files/%s_profiles_%sMass_rescaling%s.pickle" % (self.fileroot[:-3], self.model, self.model, mass_bin, self.rescaling)
        profile_dumpfile = "%s%s/pickle_files/%s_profiles_%sMass_rescaling%s_s%d.pickle" % (self.fileroot[:-3], self.model, self.model, mass_bin, self.rescaling, self.snapshot)

        mass_bins = np.linspace(12.33, 14., 6)   # 5 bins (Msun)

        # apply relevant rescalings
        mtrue = self.group_m500[self.sample]   # Msun
        t_actual = self.mass_temp_profile
        rho_actual = self.density_profile
        pressure = self.electron_pressure_profile
        if self.model[0] == "F" and self.snapshot == 45:
            # f(R) gravity at z=0 includes effective data
            meff = self.group_meff500[self.sample]
            mass_ratio = meff / mtrue
            if self.rescaling == "true":
                print("Applying 'true' rescaling")
                mass = mtrue
                t_pred = t_actual / mass_ratio[:, None]
                rho_pred = rho_actual
            else:
                print("Applying 'effective' rescaling")
                mass = meff
                t_pred = t_actual
                rho_pred = rho_actual * mass_ratio[:, None]
        elif self.model[0] == "N":
            # nDGP
            mass = mtrue
            if self.model == "N1":
                t_pred = t_actual / 1.05
            else:
                t_pred = t_actual
            rho_pred = rho_actual
        else:
            # f(R) gravity (with z>0) or GR
            mass = mtrue   # Msun
            t_pred = t_actual
            rho_pred = rho_actual

        r200 = self.group_r200[self.sample]   # kpc
        r500 = self.group_r500[self.sample]   # kpc

        if mass_bin == "xx-low":
            bin_id = 0
        if mass_bin == "x-low":
            bin_id = 1
        if mass_bin == "low":
            bin_id = 2
        if mass_bin == "middle":
            bin_id = 3
        if mass_bin == "high":
            bin_id = 4
        bin_haloes = np.where((mass > 10**(mass_bins[bin_id])) & (mass < 10**(mass_bins[bin_id+1])))

        min_rad = min(r200[bin_haloes])
        num_bins = len(np.where(self.bin_radii < min_rad)[0])

        # calculate median profile
        av_t = np.median(t_actual[bin_haloes], axis=0)[:num_bins]
        av_t_pred = np.median(t_pred[bin_haloes], axis=0)[:num_bins]
        av_rho = np.median(rho_actual[bin_haloes], axis=0)[:num_bins]
        av_rho_pred = np.median(rho_pred[bin_haloes], axis=0)[:num_bins]
#        av_pressure = np.median(pressure[bin_haloes], axis=0)[:num_bins]
        print("# haloes in profile bin: ", len(rho_actual[bin_haloes]))

        ### Testing ###
        print(self.group_m500[self.sample][bin_haloes])
        print(self.group_pos[self.sample][bin_haloes])
        pressure = list(pressure[bin_haloes])
        positions = list(self.group_pos[self.sample][bin_haloes])
        remove = []
        if self.non_rad:
            if self.snapshot == 9 and mass_bin == "low":
                remove = [9, 6, 5]
                if self.model == "F6":
                    remove.append(3)
                else:
                    remove.append(4)
            if self.snapshot == 22 and mass_bin == "middle":
                if self.model == "GR":
                    remove = [11, 3, 1]
                if self.model == "F6":
                    remove = [11, 2, 1]
                if self.model == "F5":
                    remove = [14, 13, 12, 11, 2, 1]
            if self.snapshot == 45 and mass_bin == "high":
                if self.model == "GR":
                    remove = [13, 6, 5, 2, 1, 0]
                if self.model == "F6":
                    remove = [12, 5, 4, 1, 0]
                if self.model == "F5":
                    remove = [14, 13, 12, 11, 9, 3, 2]
        else:
            if self.simulation == "L62_N512_nDGP":
                # L62 nDGP sim
                if self.snapshot == 50 and mass_bin == "middle":
                    if self.model == "N1":
                        #remove = [6, 2]
                        remove = [6, 2]
                    #remove.append(1)
                if self.snapshot == 99 and mass_bin == "high":
                    if self.model == "N5":
                        #remove = [6, 4, 2, 1, 0]
                        remove = [1]
                    if self.model == "N1":
                        #remove = [7, 6, 4, 1, 0]
                        remove = [7]
                    #if self.model == "GR":
                        #remove = [4, 3, 1, 0]
            else:
                # L62 f(R) sim
                if self.snapshot == 22 and mass_bin == "middle":
                    if self.model == "F5":
                        remove = [14, 13, 12, 11, 2]
                    remove.append(1)
                if self.snapshot == 45 and mass_bin == "high":
                    if self.model == "F5":
                        remove = [15, 14, 13, 12, 11, 5, 3, 2]
                    if self.model == "F6":
                        remove = [6, 4, 2, 1, 0]
                    if self.model == "GR":
                        remove = [5, 3, 2, 1, 0]
        for i in remove:
            pressure.pop(i)
            positions.pop(i)
        pressure = np.array(pressure)
        av_pressure = np.median(pressure, axis=0)[:num_bins]
        print("Positions: ", np.array(positions))
        print("Pressure: ", av_pressure)
        ### End testing ###

        median_r500 = np.median(np.log10(r500[bin_haloes]))
        mean_r500 = np.mean(np.log10(r500[bin_haloes]))

        df = open(profile_dumpfile,"w+b")
        pickle.dump((self.bin_radii[:num_bins], av_t, av_t_pred, av_rho, av_rho_pred, av_pressure, median_r500, mean_r500), df)
        df.close()


    def proxy_scaling_relation(self, proxy_type="SZ", no_core=False, temp_weight="mass", use_analytical=False):
        if no_core:
            # temp and Ysz exclude core region
            core_name = "no_core%s" % (self.core_label)   # unique label if core != 0.15*R500
            Ysz = self.Ysz_no_core
            Lx = self.Lx_no_core
            if temp_weight == "mass":   # mass-weighted temp
                Tgas = self.mass_T500
            elif temp_weight == "volume":   # volume-weighted temp
                Tgas = self.vol_T500

        else:
            # temp and Y include core region
            core_name = "with_core"
            Ysz = self.Ysz_with_core
            Lx = self.Lx_with_core
            if temp_weight == "mass":   # mass-weighted temp
                Tgas = self.mass_T500_with_core
            elif temp_weight == "volume":   # volume-weighted temp
                Tgas = self.vol_T500_with_core
 
        if proxy_type == "SZ":
            proxy = Ysz
            proxy_name = "ysz"
 
        if proxy_type == "Yx":   # use Yx
            # this depends on mass-weighted temperature
            proxy = self.Mg500 * self.mass_T500   # Yx, units Msun keV
            # T500 depends on core vs no-core
            proxy_name = "yx"

        if proxy_type == "Lx":
            proxy = Lx
            proxy_name = "lx"

        if proxy_type == "T":
            proxy = Tgas
            proxy_name = "tgas"

        if use_analytical:
            analytical_label = "TANH"
        else:
            analytical_label = ""

        # True mass
        mtrue = self.group_m500[self.sample]   # Msun

        # Compute effective mass only if f(R) gravity
        if not self.gr:
            if self.model[0] == "N":
                # nDGP
                meff = mtrue   # Msun
                if self.model == "N1":
                    #Apply 1/1.05 rescaling for temperature-dependence
                    if proxy_type == "Lx":
                        prediction = proxy / (1.05**0.5)
                    else:
                        prediction = proxy / 1.05
                else:
                    #No rescaling required for weaker nDGP models
                    prediction = proxy
            else:
                # f(R) gravity
                if use_analytical:
                    mass_ratio = self.correction(mtrue)
                    meff = mtrue * mass_ratio
                else:
                    meff = self.group_meff500[self.sample]
                    mass_ratio = meff / mtrue

                if self.rescaling == "true":
                    print("Applying 'true' rescaling")
                    if proxy_type == "Lx":
                        prediction = proxy / (mass_ratio**0.5)
                    else:
                        prediction = proxy / mass_ratio
                else:
                    print("Applying 'effective' rescaling")
                    if proxy_type == "Lx":
                        prediction = proxy * (mass_ratio)**2
                    elif proxy_type == "T":
                        prediction = proxy
                    else:
                        prediction = proxy * mass_ratio

        # name dumpfile for binned data

        ### For referee ###
        gr_dumpfile = self.fileroot[:-3] + "GR/pickle_files/sfr_%s_gr_%s_%s_weight_%s_s%d.pickle" % (proxy_name, core_name, temp_weight, self.file_ending, self.snapshot)
        ### End ###
#        gr_dumpfile = self.fileroot[:-3] + "GR/pickle_files/%s_gr_%s_%s_weight_%s_s%d.pickle" % (proxy_name, core_name, temp_weight, self.file_ending, self.snapshot)

        fr_dumpfile = self.fileroot[:-3] + "%s/pickle_files/%s_%s_%s_%s_weight_%s_%srescaling%s_s%d.pickle" % (self.model, proxy_name, self.model, core_name, temp_weight, self.file_ending, analytical_label, self.rescaling, self.snapshot)

        # Store high-mass data (exceeding 4e12 Msun/h)
        high_mass = np.where((mtrue > 4e12 / 0.6774))
        if self.gr:
            df = open(gr_dumpfile,"w+")
            pickle.dump((mtrue[high_mass], proxy[high_mass]), df)
            df.close()

        else:
            df = open(fr_dumpfile,"w+")
            pickle.dump((meff[high_mass], mtrue[high_mass], proxy[high_mass], prediction[high_mass]), df)
            df.close()



    def correction(self, m):
        '''Enhancement of the dynamical mass with respect to the true mass,
        given the true mass in units Msun, for z=0.
        '''
        p1=2.21
        if self.model=="F5":
            f_R0=1.e-5
        elif self.model=="F6":
            f_R0=1.e-6
        fR = f_R(f_R0, 1 / (1. + self.s.header.redshift), 0.3089, 0.6911)
        p2 = 1.503 * np.log10(fR / (1. + self.s.header.redshift)) + 21.64  # 10^p2 units Msun/0.697
        p2_rescaled = np.log10(10**p2 / 0.697)   # 10^p2' units Msun
        return 7 / 6 - (1 / 6) * np.tanh(p1 * (np.log10(m) - p2_rescaled))



class LoadDumpfile:
    ''' Load group properties from a specified pickle dumpfile, if this exists.
    '''
    def __init__(self, group_dumpfile, subhalo_dumpfile, non_rad=False): 
        if os.path.exists(group_dumpfile):
            print("%s exists!" % (group_dumpfile))
            df = open(group_dumpfile, 'rb')
            if non_rad:                
                (self.M500, self.M200, self.Mg500, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas) = pickle.load(df)
            else:
                (self.M500, self.M200, self.Mg500, self.Mstar, self.A19_Mstar, self.SMF, self.vol_T500, self.mass_T500, self.vol_T500_with_core, self.mass_T500_with_core, self.Ysz_with_core, self.Ysz_no_core, self.Lx_with_core, self.Lx_no_core, self.vol_temp_profile, self.mass_temp_profile, self.density_profile, self.electron_pressure_profile, self.cum_fgas) = pickle.load(df)
            df.close()

        else:
            print("%s does not exist!" % (group_dumpfile))
            sys.exit(0)

        if os.path.exists(subhalo_dumpfile):
            print("%s exists!" % (subhalo_dumpfile))
            df = open(subhalo_dumpfile, 'rb')
            (self.sub_starmass) = pickle.load(df)
            df.close()

        else:
            print("%s does not exist!" % (subhalo_dumpfile))
            sys.exit(0)




class SimulationTesting:
    def __init__(self, simulation, models, realisations, snapshot, file_ending, labels, colors, defaults, plot_name, system='cosma7', show_spread=False):
        self.simulation = simulation
        self.models = models
        self.realisations = realisations
        self.snapshot = snapshot
        self.file_ending = file_ending
#       self.labels = [r'${\rm t_{star}}$: 2.27', r'${\rm t_{star}}$: 1.82', r'${\rm t_{star}}$: 1.59', r'${\rm t_{star}}$: 1.36', r'${\rm t_{star}}$: 1.14']
        self.labels = labels
        self.colors = colors
        self.defaults = defaults
        self.plot_name = plot_name
        self.system = system
        self.show_spread = show_spread

        # load snapshot for a realisation (for h, Omega, etc)
        if simulation == "L100_N256":
            if system[0] == "cosma7":
                fd = FindData(simulation,models[0],"Wind_1",snapshot,system=system[0])
            elif system[0] == "madfs":
                fd = FindData(simulation,models[0],"Wind_14",snapshot,system=system[0])
        elif simulation == "L54_N256":
            fd = FindData(simulation,models[0],"1",snapshot,system=system[0])
        elif simulation == "L86_N256":
            if system[0] == "cosma7":
                fd = FindData(simulation,models[0],"RhoWindBH_11",snapshot,system=system[0])
            elif system[0] == "madfs":
                fd = FindData(simulation,models[0],"1",snapshot,system=system[0])
        elif simulation == "L68_N256":
            if system[0] == "cosma7":
                fd = FindData(simulation,models[0],"RhoWind_1",snapshot,system=system[0])
            elif system[0] == "madfs":
                fd = FindData(simulation,models[0],"1",snapshot,system=system[0])
            elif system[0] == "cosma6":
                fd = FindData(simulation,models[0],"RhoWind_10",snapshot,system=system[0])
        elif simulation == "L136_N512":
            fd = FindData(simulation,models[0],"RhoWind_3",snapshot,system=system[0])
        elif simulation == "L302_N1136":
            fd = FindData(simulation,"GR","1",snapshot,system=system[0])
        elif simulation == "L62_N512":
            fd = FindData(simulation,models[0],"1",snapshot,system=system[0])

        self.particle_path = fd.particle_path
        self.group_path = fd.group_path
        self.s = read_hdf5.snapshot(snapshot, directory=self.particle_path, dirbases=["snapdir_", ""], snapbases=["/GR_", "/gadget", "/snap_"], check_total_particle_number=True)

        self.fileroot = "/cosma8/home/dp203/dc-pick1/cluster_properties/plots/%s/" % (simulation)
        self.dumpfiles = [self.fileroot+"%s/pickle_files/%s_%s_%s_s%d_%s.pickle" % (m,simulation,m,realisations[mid],snapshot,file_ending) for (mid, m) in enumerate(models)]
        self.subhalo_dumpfiles = [self.fileroot+"%s/pickle_files/subhalo_%s_%s_%s_s%d_%s.pickle" % (m,simulation,m,realisations[mid],snapshot,file_ending) for (mid, m) in enumerate(models)]

#       self.subhalo_dumpfiles2 = [self.fileroot+"pickle_files/subhalo_100kpc_%s_%s_%s_s%d_%s.pickle" % (simulation,model,r,snapshot,file_ending) for r in realisations]



    def tng_observables(self):
        # set fontsizes for figures
        self.mysize = "large"
        self.axsize = "medium"
        self.legsize = "small"
        self.ms = 5.
        self.lw = 1.

        fig, ax = plt.subplots(3, 2, sharey=False, sharex = False, figsize=(7.25, 10.5))
        fig.subplots_adjust(left=0.083, right=0.975, bottom=0.04, top=0.995, hspace=0.15, wspace=0.2)

        self.stellar_mass_fraction(ax[0, 0])
        self.stellar_mass_function(ax[0, 1])
        self.sfrd(ax[1, 0])
        self.gas_mass_fraction(ax[1, 1])
        self.black_hole_mass(ax[2, 0])
        self.galaxy_size(ax[2, 1])

        lns = [] 
        for lid, l in enumerate(self.labels):
            lns += ax[0, 1].plot([], [], linewidth=self.lw, color=self.colors[lid], label=l)
        labs = [l.get_label() for l in lns]
        leg2 = ax[0, 1].legend(lns, labs, loc='upper right', fontsize=self.legsize, frameon=False)

        for did, d in enumerate(self.defaults):
            ax[0, 1].annotate(d, xy=(0.97, 0.90-(did*0.05)), xycoords='axes fraction', fontsize = self.legsize, horizontalalignment='right', verticalalignment='top')

        fig.savefig(self.fileroot+"observables/%s.pdf" % (self.plot_name))


    def gas_mass_fraction(self, ax):
#       obs_data = np.genfromtxt("/cosma7/data/dp004/dc-mitc2/hydro_analysis/obs_data/fgas_obs_data.txt")   # from Ian McCarthy
#       print("References: ", obs_data[:, 6])
#       sun_args = np.where((obs_data[:, 6] == "Sun2009"))
#       pratt_args = np.where((obs_data[:, 6] == "Pratt2009"))
#       lovisari_args = np.where((obs_data[:, 6] == "Lovisari2015"))
#       gonzalez_args = np.where((obs_data[:, 6] == "Gonzalez2013"))
#       obs_mass = obs_data[:,0]*1e13
#       obs_fgas = obs_data[:,3]
        obs_data = np.genfromtxt("/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/lovisari_obs_data.txt")
        ax.scatter(obs_data[:,0] * 1e13, obs_data[:,3], marker='^', s=self.ms, color="gray", label=r'Lovisari+2015')
        obs_data = np.genfromtxt("/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/gonzalez_obs_data.txt")
        ax.scatter(obs_data[:,0] * 1e13, obs_data[:,3], marker='>', s=self.ms, color="gray", label=r'Gonzalez+2013')
        obs_data = np.genfromtxt("/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/pratt_obs_data.txt")
        ax.scatter(obs_data[:,0] * 1e13, obs_data[:,3], marker='s', s=self.ms, color="gray", label=r'Pratt+2010')
        obs_data = np.genfromtxt("/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/sun_obs_data.txt")
        ax.scatter(obs_data[:,0] * 1e13, obs_data[:,3], marker='o', s=self.ms, color="gray", label=r'Sun+2009')
#        ax.scatter(obs_mass, obs_fgas, s=self.ms, color="gray", label=r'Observations')

        fgas_TNG25 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng25_gas_fraction.txt')
        ax.plot(np.power(10, fgas_TNG25[:,0]), fgas_TNG25[:,1], ls = '-.', linewidth = self.lw, color = 'black', label = r'TNG L25-N512')

        
        for (mid, m) in enumerate(self.models):
            ld = LoadDumpfile(self.dumpfiles[mid], self.subhalo_dumpfiles[mid])
            fraction = ld.Mg500 / ld.M500
            mass = ld.M500   # M500c defn to compare with obs
            print("%d haloes with mass greater than 10^14 Msun" % (len(mass[mass > 1.e14])))
            print("%d haloes with mass greater than 10^15 Msun" % (len(mass[mass > 1.e15])))
            bins = np.logspace(np.log10(1.e11), np.log10(max(mass)+1), 18, base=10.0)
            digitized = np.digitize(mass, bins)
            mean_log_mass = np.array([10**(float(np.mean(np.log10(mass[digitized == i])))) for i in range(1, len(bins))])   # units Msun
            median_fraction = np.array([float(np.median(fraction[digitized == i])) for i in range(1, len(bins))])
            size = np.array([len(fraction[digitized == i]) for i in range(1, len(bins))])
            ax.plot(mean_log_mass[size>=5], median_fraction[size>=5], linewidth=self.lw, color=self.colors[mid])
            if self.show_spread and m == "GR":
                upper_percentile = np.array([float(np.percentile(fraction[digitized == i], 84)) for i in range(1, len(bins))])
                lower_percentile = np.array([float(np.percentile(fraction[digitized == i], 16)) for i in range(1, len(bins))])
                ax.fill_between(mean_log_mass[size>=5], lower_percentile[size>=5], upper_percentile[size>=5], facecolor=self.colors[mid], alpha=0.1)

        ax.set_xscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([1e11, 1e15])
        ax.set_ylim([0.01, 0.17])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.linspace(0.05,0.15,3))
        ax.set_yticklabels([r'$0.05$',r'$0.10$',r'$0.15$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.logspace(11,15,5,base=10.0))
        ax.set_xticklabels([r'$10^{11}$',r'$10^{12}$',r'$10^{13}$',r'$10^{14}$','$10^{15}$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc=2, fontsize=self.legsize, frameon=False)
        ax.set_xlabel(r'$M_{\rm 500c}$ $[M_{\odot}]$', fontsize=self.mysize)
        ax.set_ylabel(r'$[M_{\rm gas}/M_{\rm tot}]_{\rm 500c}$', fontsize=self.mysize)


    def stellar_mass_fraction(self, ax):
        stellar_mass_behroozi = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/behroozi2013_sm_hm.txt')
        yerr = (np.array([stellar_mass_behroozi[:,4], stellar_mass_behroozi[:,5]])) * (0.3089/0.0486)
        mfrac_TNG_L25_N128 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/TNG_L25_N128_mfrac.txt')
        mfrac_TNG_L25_N512 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/TNG_L25_N512_mfrac.txt')
        mfrac_bahamas = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/bahamas_smhm.txt')
        mfrac_TNG100 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng100_smhm.txt')
        mfrac_TNG300 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng300_smhm.txt')

        ax.plot(stellar_mass_behroozi[:,0], stellar_mass_behroozi[:,1]*(0.3089/0.0486), ls='-', linewidth = self.lw, color='darkgray', label='Behroozi+2013')
        ax.fill_between(stellar_mass_behroozi[:,0], stellar_mass_behroozi[:,1]*(0.3089/0.0486) - yerr[0], stellar_mass_behroozi[:,1]*(0.3089/0.0486) + yerr[1], facecolor='darkgray', alpha=0.35)        

        stellar_mass_kravtsov = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/kravtsov18_smf.txt')
        ax.plot(np.power(10, stellar_mass_kravtsov[:, 0]), stellar_mass_kravtsov[:, 1], ls='-', linewidth = self.lw, color='lightgray', label='Kravtsov+2018')
        ax.fill_between(np.power(10, stellar_mass_kravtsov[:, 0]), stellar_mass_kravtsov[:, 1] - stellar_mass_kravtsov[:, 2], stellar_mass_kravtsov[:, 1] + stellar_mass_kravtsov[:, 3], facecolor='lightgray', alpha=0.35)
       
        ax.plot(mfrac_TNG_L25_N512[:,0], mfrac_TNG_L25_N512[:,1], ls = '-.', linewidth = self.lw, color = 'black', label = r'TNG L25-N512') 
        ax.plot(np.power(10, mfrac_TNG100[:,0]), mfrac_TNG100[:,1], ls = 'solid', linewidth = self.lw, color = 'black', label = r'TNG100')
        ax.plot(np.power(10, mfrac_TNG300[:,0]), mfrac_TNG300[:,1], ls = 'dashed', linewidth = self.lw, color = 'black', label = r'TNG300')
        ax.plot(np.power(10, mfrac_bahamas[:,0]), mfrac_bahamas[:,1] * 0.3089 / 0.0486, ls = 'dotted', linewidth = self.lw, color = 'black', label = r'BAHAMAS')

        for (mid,m) in enumerate(self.models):
            ld = LoadDumpfile(self.dumpfiles[mid], self.subhalo_dumpfiles[mid])
            smf = ld.SMF
            mass = ld.M200
            bins = np.logspace(np.log10(8e10), np.log10(max(mass)+1), 15, base=10.0)
            digitized = np.digitize(mass, bins)
            mean_log_mass = np.array([10**(float(np.mean(np.log10(mass[digitized == i])))) for i in range(1, len(bins))])   # units Msun
            median_smf = np.array([float(np.median(smf[digitized == i])) for i in range(1, len(bins))])
            mean_smf = np.array([float(np.mean(smf[digitized == i])) for i in range(1, len(bins))])
            upper_error = np.array([float(np.percentile(smf[digitized == i], 84)) for i in range(1, len(bins))]) - median_smf
            lower_error = median_smf - np.array([float(np.percentile(smf[digitized == i], 16)) for i in range(1, len(bins))])
            size = np.array([len(smf[digitized == i]) for i in range(1, len(bins))])
            ax.plot(mean_log_mass[size>=5], mean_smf[size>=5] * 0.3089 / 0.0486, linewidth=self.lw, color=self.colors[mid])
            if self.show_spread and m == "GR":
                ax.fill_between(mean_log_mass[size>=5], (mean_smf[size>=5] - lower_error[size>=5]) * 0.3089 / 0.0486, (mean_smf[size>=5] + upper_error[size>=5]) * 0.3089 / 0.0486, facecolor=self.colors[mid], alpha=0.1)

        ax.set_xscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([1e11, 1e15])
        ax.set_ylim([0.00, 0.25])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.linspace(0.0,0.2,3))
        ax.set_yticklabels([r'$0.0$',r'$0.1$',r'$0.2$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.logspace(12,15,4,base=10.0))
        ax.set_xticklabels([r'$10^{12}$',r'$10^{13}$',r'$10^{14}$',r'$10^{15}$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc=1, fontsize=self.legsize, frameon=False)
        ax.set_xlabel(r'$M_{\rm 200c}$ $[M_{\odot}]$', fontsize=self.mysize)
        ax.set_ylabel(r'$M_{\star}/M_{\rm halo}$ * $(\Omega_{\rm M}/\Omega_{\rm b})$', fontsize=self.mysize)


    def stellar_mass_function(self, ax):
        my_smf_dsouza2015 = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/dsouza2015_smfunction.txt')
        my_smf_bernardi2013 = np.genfromtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/bernardi2013_smfunction_cmodel.dat')
        my_smf_baldry2012 = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/baldry2012_smfunction.txt')
        my_smf_li_and_white2009 = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/li_white.txt')
        smf_TNG_L25_N512 = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/TNG_L25_N512_smf.txt')
        smf_TNG100 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng100_phi_correct.txt')
        smf_bahamas = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/bahamas_smf.txt')
        smf_cosmo_owls = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/cosmo_owls.txt')

        hubble = 0.72

        ax.scatter(10**(my_smf_dsouza2015[:,0]) / (hubble**2), 10**(my_smf_dsouza2015[:,1]) * (hubble**3), marker = '^', s=self.ms, color = 'gray', label = 'D\'Souza+2015')
        ax.errorbar(10**(my_smf_dsouza2015[:,0]) / (hubble**2), 10**(my_smf_dsouza2015[:,1]) * (hubble**3), yerr=((10**(my_smf_dsouza2015[:,1])-10**(my_smf_dsouza2015[:,1]-my_smf_dsouza2015[:,4])) * (hubble**3), (10**(my_smf_dsouza2015[:,1]+my_smf_dsouza2015[:,5])-10**(my_smf_dsouza2015[:,1])) * (hubble**3)), ls = 'none', elinewidth = self.lw, color = 'gray')

        ax.scatter(10**(my_smf_bernardi2013[:,0]), 10**(my_smf_bernardi2013[:,1]), marker='o', s=self.ms, color='gray', label = r'Bernardi+2013')
        ax.errorbar(10**(my_smf_bernardi2013[:,0]), 10**(my_smf_bernardi2013[:,1]), yerr=(10**(my_smf_bernardi2013[:,1])-10**(my_smf_bernardi2013[:,1]-my_smf_bernardi2013[:,2]), 10**(my_smf_bernardi2013[:,1]+my_smf_bernardi2013[:,2])-10**(my_smf_bernardi2013[:,1])), ls = 'none', elinewidth = self.lw, color = 'gray')

        ax.scatter(my_smf_baldry2012[:,0], my_smf_baldry2012[:,1], marker='s', s=self.ms, color='gray', label=r'Baldry+2012')
        ax.errorbar(my_smf_baldry2012[:,0], my_smf_baldry2012[:,1], yerr=my_smf_baldry2012[:,2], ls = 'none', elinewidth = self.lw, color = 'gray')

        hubble = 0.7
        upper = np.power(10, my_smf_li_and_white2009[:,1] + my_smf_li_and_white2009[:,5] + (3 * np.log10(hubble)))
        lower = np.power(10, my_smf_li_and_white2009[:,1] - my_smf_li_and_white2009[:,4] + (3 * np.log10(hubble)))
        upper_err = upper - (np.power(10, my_smf_li_and_white2009[:,1] + (3 * np.log10(hubble))))
        lower_err = (np.power(10, my_smf_li_and_white2009[:,1] + (3 * np.log10(hubble)))) - lower

        ax.scatter(np.power(10, my_smf_li_and_white2009[:,0] - (2 * np.log10(hubble))), np.power(10, my_smf_li_and_white2009[:,1] + (3 * np.log10(hubble))), marker='>', s=self.ms, color='gray', label=r'Li \& White 2009')

        ax.errorbar(np.power(10, my_smf_li_and_white2009[:,0] - (2 * np.log10(hubble))), np.power(10, my_smf_li_and_white2009[:,1] + (3 * np.log10(hubble))), yerr=(lower_err, upper_err), ls = 'none', elinewidth = self.lw, color = 'gray')

        ax.plot(smf_TNG_L25_N512[:,0], smf_TNG_L25_N512[:,1], ls = '-.', linewidth=self.lw, color = 'black', label=r'TNG L25-N512')
        ax.plot(np.power(10, smf_TNG100[:,0]), np.power(10, smf_TNG100[:,1]), ls = 'solid', linewidth=self.lw, color = 'black', label=r'TNG100')
        ax.plot(np.power(10, smf_bahamas[:,0]), smf_bahamas[:,1], ls = 'dotted', linewidth=self.lw, color = 'black', label=r'BAHAMAS')
        ax.plot(np.power(10, smf_cosmo_owls[:,0]), np.power(10, smf_cosmo_owls[:,1]), ls = 'dashed', linewidth=self.lw, color = 'black', label=r'cosmo-OWLS (AGN 8.5)')

        for (mid,m) in enumerate(self.models):
            ld = LoadDumpfile(self.dumpfiles[mid], self.subhalo_dumpfiles[mid])
            mass = np.log10(ld.sub_starmass)
            bins = np.linspace(9.8, max(mass) + 0.0001, 11)
            digitized = np.digitize(mass, bins)
            mean_log_mass = np.array([10 ** (float(np.mean(mass[digitized == i]))) for i in range(1, len(bins))])  # units Msun
            size = np.array([len(mass[digitized == i]) for i in range(1, len(bins))])        
            diff = bins[1:]-bins[:-1]
            sm_function = size / ((self.s.header.boxsize / 1000 / self.s.header.hubble) ** 3 * diff)

            ax.plot(mean_log_mass[size >= 5], sm_function[size >= 5], linewidth=self.lw, color=self.colors[mid])

            if self.show_spread and m == "GR":
                # Compute JK uncertainties
                fd = FindData(self.simulation, m, self.realisations[mid], self.snapshot, system = self.system[mid])
                s = read_hdf5.snapshot(self.snapshot, directory = fd.particle_path, dirbases = ["snapdir_", ""], snapbases = ["/GR_", "/gadget", "/snap_"], check_total_particle_number = True)
                s.group_catalog(['SubhaloPos'])
                scale_factor = 1 / (1. + s.header.redshift)
                sub_pos = s.cat['SubhaloPos'] * scale_factor * fd.unit_factor   # kpc
                count_errors = self.jackknife(sub_pos, mass, bins, nsub=27)
                sm_function_errors = count_errors / ((self.s.header.boxsize / 1000 / self.s.header.hubble) ** 3 * 26 / 27 * diff)   # 26 / 27 factor accounts for JK subvolume size
                ax.fill_between(mean_log_mass[size>=5], (sm_function[size>=5] - sm_function_errors[size>=5]), (sm_function[size>=5] + sm_function_errors[size>=5]), facecolor=self.colors[mid], alpha=0.1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([1e10, 1e12])
        ax.set_ylim([1e-6, 2e-2])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.logspace(-6,-2,5,base=10.0))
        ax.set_yticklabels([r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.logspace(10,12,3,base=10.0))
        ax.set_xticklabels([r'$10^{10}$',r'$10^{11}$','$10^{12}$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        leg1 = ax.legend(loc=3, fontsize=self.legsize, frameon=False)
        ax.add_artist(leg1)
        ax.set_xlabel(r'$M_{\star}$ $[<{\rm 30kpc}]$ $[M_{\odot}]$', fontsize=self.mysize)
        ax.set_ylabel(r'$\Phi$ $[{\rm Mpc^{-3} dex^{-1}}]$', fontsize=self.mysize)


    def black_hole_mass(self, ax):
        mcconnell = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/mcconnell.txt')
        ax.scatter(np.power(10, mcconnell[:,0]), np.power(10, mcconnell[:,1]), marker='s', s = self.ms, color = 'gray', label = r'McConnell \& Ma 2013 (comp)')
        bh_mass_TNG25 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng25_bh_mass.txt')
        ax.plot(np.power(10, bh_mass_TNG25[:,0]), np.power(10, bh_mass_TNG25[:,1]), ls = '-.', linewidth = self.lw, color = 'black', label = r'TNG L25-N512')

        for (rid,r) in enumerate(self.realisations):
            if self.simulation == "L302_N1136" and self.models[rid] == "F5":
                # Plot xyscan data (raw data lost during cosma8 crash)
                data = np.genfromtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/l302_f5_bh_mass.txt')
                masses = np.power(10, data[:, 0])
                bh_masses = np.power(10, data[:, 1])
                ax.plot(masses, bh_masses, linewidth=self.lw, color=self.colors[rid])
            else:
                # Compute using simulation group data
                fd = FindData(self.simulation,self.models[rid],r,self.snapshot,system=self.system[rid])
                particle_path = fd.particle_path
                s = read_hdf5.snapshot(self.snapshot, directory=particle_path, dirbases=["snapdir_", ""], snapbases=["/GR_", "/gadget", "/snap_"], check_total_particle_number=True)
                s.group_catalog(['SubhaloMassInRadType', 'SubhaloBHMass'])
                stellar_mass = s.cat['SubhaloMassInRadType'][:,4][s.cat['SubhaloBHMass'] > 0.0]   # Msun, <2r_star0.5
                bh_mass = s.cat['SubhaloBHMass'][s.cat['SubhaloBHMass'] > 0.0]*(1e10/0.6774)   # BH mass in subhalo, Msun
                bins = np.logspace(8.0, np.log10(max(stellar_mass)), 25, base=10.0)
                digitized = np.digitize(stellar_mass, bins)
                mean_log_mass = np.array([10**(float(np.mean(np.log10(stellar_mass[digitized == i])))) for i in range(1, len(bins))])   # units Msun
                median_bh_mass = np.array([float(np.median(bh_mass[digitized == i])) for i in range(1, len(bins))])
                size = np.array([len(bh_mass[digitized == i]) for i in range(1, len(bins))])
                ax.plot(mean_log_mass[size>=5], median_bh_mass[size>=5], linewidth=self.lw, color=self.colors[rid])
                if self.show_spread and self.models[rid] == "GR":
                    upper_percentile = np.array([float(np.percentile(bh_mass[digitized == i], 84)) for i in range(1, len(bins))])
                    lower_percentile = np.array([float(np.percentile(bh_mass[digitized == i], 16)) for i in range(1, len(bins))])
                    ax.fill_between(mean_log_mass[size>=5], lower_percentile[size>=5], upper_percentile[size>=5], facecolor=self.colors[rid], alpha=0.1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([1e10, 3e12])
        ax.set_ylim([8e6, 2e10])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.logspace(7,10,4,base=10.0))
        ax.set_yticklabels([r'$10^{7}$',r'$10^{8}$',r'$10^{9}$',r'$10^{10}$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.logspace(10,12,3,base=10.0))
        ax.set_xticklabels([r'$10^{10}$',r'$10^{11}$',r'$10^{12}$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc=2, fontsize=self.legsize, frameon=False)
        ax.set_xlabel(r'$M_{\star}$ $[<2r_{\star.1/2}]$ $[M_{\odot}]$', fontsize=self.mysize)
        ax.set_ylabel(r'$M_{\rm BH}$ $[M_{\odot}]$ $[{\rm all}$ ${\rm subhaloes}]$', fontsize=self.mysize)


    def galaxy_size(self, ax):
        baldry_diamonds = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/baldry_diamonds.txt')
        ax.scatter(np.power(10, baldry_diamonds[-3:,0]), np.power(10, baldry_diamonds[-3:,1]), marker='D', s = self.ms, color = 'lightgray', label = r'Baldry et al.~2012 (GAMA Re, blue)')

        baldry_circles = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/baldry_circles.txt')
        ax.scatter(np.power(10, baldry_circles[-3:,0]), np.power(10, baldry_circles[-3:,1]), marker='o', s = self.ms, color = 'darkgray', label = r'Baldry et al.~2012 (GAMA Re, red)')

        galaxy_size_TNG25 = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/tng25_galaxy_size.txt')
        ax.plot(np.power(10, galaxy_size_TNG25[:,0]), np.power(10, galaxy_size_TNG25[:,1]), ls = '-.', linewidth = self.lw, color = 'black', label = r'TNG L25-N512')
        
        for (rid,r) in enumerate(self.realisations):
            if self.simulation == "L302_N1136" and self.models[rid] == "F5":
                # Plot xyscan data (raw data lost during cosma8 crash)
                data = np.genfromtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/l302_f5_galaxy_size.txt')
                masses = np.power(10, data[:, 0])
                sizes = np.power(10, data[:, 1])
                ax.plot(masses, sizes, linewidth=self.lw, color=self.colors[rid])
            else:
                # Compute using simulation group data
                fd = FindData(self.simulation,self.models[rid],r,self.snapshot,system=self.system[rid])
                particle_path = fd.particle_path
                s = read_hdf5.snapshot(self.snapshot, directory=particle_path, dirbases=["snapdir_", ""], snapbases=["/GR_", "/gadget", "/snap_"], check_total_particle_number=True)
                scale_factor = 1 / (1. + s.header.redshift)
                s.group_catalog(['SubhaloMassInRadType', 'SubhaloHalfmassRadType'])
                stellar_mass = s.cat['SubhaloMassInRadType'][:,4]   # Msun, <2r_star0.5
                stellar_radius = s.cat['SubhaloHalfmassRadType'][:,4] * scale_factor   # kpc, r_star0.5
                bins = np.logspace(8.0, np.log10(max(stellar_mass)), 25, base=10.0)
                digitized = np.digitize(stellar_mass, bins)
                mean_log_mass = np.array([10**(float(np.mean(np.log10(stellar_mass[digitized == i])))) for i in range(1, len(bins))])   # units Msun
                median_radius = np.array([float(np.median(stellar_radius[digitized == i])) for i in range(1, len(bins))])
                size = np.array([len(stellar_radius[digitized == i]) for i in range(1, len(bins))])
                ax.plot(mean_log_mass[size>=5], median_radius[size>=5], linewidth=self.lw, color=self.colors[rid])
                if self.show_spread and self.models[rid] == "GR":
                    upper_percentile = np.array([float(np.percentile(stellar_radius[digitized == i], 84)) for i in range(1, len(bins))])
                    lower_percentile = np.array([float(np.percentile(stellar_radius[digitized == i], 16)) for i in range(1, len(bins))])
                    ax.fill_between(mean_log_mass[size>=5], lower_percentile[size>=5], upper_percentile[size>=5], facecolor=self.colors[rid], alpha=0.1)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([1e10, 3e12])
        ax.set_ylim([2.3e0, 3e2])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.logspace(1,2,2,base=10.0))
        ax.set_yticklabels([r'$10^{1}$',r'$10^{2}$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.logspace(10,12,3,base=10.0))
        ax.set_xticklabels([r'$10^{10}$',r'$10^{11}$',r'$10^{12}$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc=2, fontsize=self.legsize, frameon=False)
        ax.set_xlabel(r'$M_{\star}$ $[<2r_{\star.1/2}]$ $[M_{\odot}]$', fontsize=self.mysize)
        ax.set_ylabel(r'$r_{\star.1/2}$ $[{\rm kpc}]$ $[{\rm all}$ ${\rm subhaloes}]$', fontsize=self.mysize)


    def sfrd(self, ax):
        sfrd_behroozi  = np.loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/behroozi2013.txt')
        sfrd_TNG_L25_N128  = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/TNG_L25_N128_sfrd.txt')
        sfrd_TNG_L25_N512  = loadtxt('/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/obs_data/TNG_L25_N512_sfrd.txt')
        ax.scatter(sfrd_behroozi[:,0], sfrd_behroozi[:,1], color = 'dimgrey', marker = 'o', s = self.ms, label = r'Behroozi+2013')
        ax.plot(sfrd_TNG_L25_N512[:,0], sfrd_TNG_L25_N512[:,1], ls = '-.', linewidth = self.lw, color = 'black', label=r'TNG L25-N512')

        zbins = np.concatenate((np.linspace(0, 1, 20), np.logspace(0.01, 1, 70)))

        for (rid,r) in enumerate(self.realisations):
            fd = FindData(self.simulation,self.models[rid],r,self.snapshot,system=self.system[rid])
            data_path = fd.particle_path
            sfr_data = np.loadtxt(data_path+'sfr.txt')
            z = 1. / sfr_data[:,0] - 1.
            sfr, edges, binnum = stat.binned_statistic(z, sfr_data[:,3], bins=zbins, statistic = 'mean')   # mean sfr measured for each z bin
            redshifts = (edges[1:] + edges[:-1]) / 2.
            sfrd = sfr / (((self.s.header.boxsize/1000)/self.s.header.hubble)**3)   # star formation rate density in Msun yr^-1 Mpc^-3
            #if self.models[rid] == "GR":
            #    ax.plot(redshifts, sfrd, linewidth=self.lw, color=self.colors[rid])    # This allows plotting of less models
            ax.plot(redshifts, sfrd, linewidth=self.lw, color=self.colors[rid])

            #if self.show_spread and self.models[rid] == "GR":
            #    error_file = "/cosma7/data/dp004/dc-mitc2/hydro_analysis/obs_data/l302_gr_sfrd.pickle"
            #    df = open(error_file, "r")
            #    (jk_redshifts, jk_sfrd, jk_sfrd_errors) = pickle.load(df)
            #    df.close()

            #    ax.plot(jk_redshifts, jk_sfrd, linewidth=self.lw, color='black')
            #    ax.fill_between(jk_redshifts, (jk_sfrd - jk_sfrd_errors), (jk_sfrd + jk_sfrd_errors), facecolor=self.colors[rid], alpha=0.5)

        ax.set_yscale('log')
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_xlim([0, 10])
        ax.set_ylim([3e-3, 2e-1])
        ax.set_yticklabels(r'')
        ax.set_yticks(np.logspace(-2,-1,2,base=10.0))
        ax.set_yticklabels([r'$10^{-2}$',r'$10^{-1}$'],fontsize=self.axsize)
        ax.set_xticklabels(r'')
        ax.set_xticks(np.linspace(0,10,6))
        ax.set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$'],fontsize=self.axsize)
        [i.set_linewidth(1.5) for i in ax.spines.values()]
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc=1, fontsize=self.legsize, frameon=False)
        ax.set_xlabel(r'$z$', fontsize=self.mysize)
        ax.set_ylabel(r'${\rm SFRD}$ $[M_{\odot}{\rm yr^{-1}Mpc^{-3}}]$', fontsize=self.mysize)


    def jackknife(self, pos, mass, bins, nsub):
        """Performs jackknife resampling of input halo data 
        (logs of mass and a mass proxy), evaluating the 1sigma 
        uncertainty of the mean mass and median proxy. 

        This is achieved by: 
        -> splitting the simulation box into nsub sub-volumes; 
        -> systematically remove haloes of one sub-volume at a time 
           to generate nsub jackknife samples; 
        -> apply same mass binning to haloes of each sample, to 
           generate nsub jackknife replicates of the median proxy and 
           mean mass of each bin; 
        -> use sqrt of diagonals of covariance matrix (with nsub-1 prefactor) 
           to predict the standard deviation.
        """
        side_n = nsub ** (1/3)   # number of sub-volumes along a side
        print("Box size [kpc / h]: ", self.s.header.boxsize)
        cell_size = self.s.header.boxsize / self.s.header.hubble / side_n   # units kpc
        edges = [i * cell_size for i in range(int(side_n) + 1)]
        group_id = np.zeros(len(mass)) - 1
        cell_id = 0   # cell IDs span 0 to nsub - 1
        group_num = 0
        group_sum = 0

        print("")
        print("Dividing %d subhaloes between %d sub-volumes..." % (len(mass), nsub))
        for k in range(int(side_n)):
            for j in range(int(side_n)):
                for i in range(int(side_n)):
                    # check which groups are in cell (i,j,k)
                    for index in range(len(mass)):
                        if edges[i] < pos[index][0] < edges[i+1] and edges[j] < pos[index][1] < edges[j+1] and edges[k] < pos[index][2] < edges[k+1]:
                            group_id[index] = cell_id
                            group_num += 1
                    print("%d subhaloes in cell %d" % (group_num, cell_id + 1))
                    group_sum += group_num

                    # Move on to next cell
                    group_num = 0
                    cell_id += 1

        print("%d groups assigned to sub-volumes" % (group_sum))

        binned_count = np.zeros((nsub, len(bins) - 1))
        for i in range(nsub):
            # create nsub JK samples by systematically removing one cell each
            mass_sample = mass[group_id != i]
            print("%d haloes in JK sample %d" % (len(mass_sample), i+1))

            # binning of proxy and mass data
            digitized = np.digitize(mass_sample, bins)
            binned_count[i] = np.array([len(mass_sample[digitized == j]) for j in range(1, len(bins))])

        print("Binned counts: ", binned_count)
 
        # Average of the nsub JK replicates for each bin
        average_count = np.mean(binned_count, axis=0)

        count_error = np.zeros(len(bins) - 1)
        for j in range(len(bins) - 1):
            count_error[j] = np.sqrt(((nsub - 1) / nsub) * np.sum((binned_count[:,j] - average_count[j])**2))

        print("Count error: ", count_error)
        return count_error



#if __name__ == "__main__":
#    simulation = "L302_N1136"
#    simulation = "L68_N256"
    # e.g., "TNG300-3", "L62_N512", "L62_N512_non_rad", "L25_N512, "L86_N256", "L62_N512_nDGP"
#    models = ["GR","F60","F55","F50","F45","F40"]
#    models = ["GR", "GR", "GR", "GR", "GR"]
#    realisations = ["1", "1"]
#    realisations = ["1", "Soft_1", "WindSoft_2", "RhoWind_2_updated", "RhoWindBH_7"]
#    system = ["cosma8", "cosma6"]
#    system = ["madfs", "madfs", "madfs", "cosma7", "cosma6"]
#    plot_labels = [r'L302-N1136-F5', r'L302-N1136-GR']
#    plot_labels = [r'TNG params', r'Soft $1/20$', r'$+\bar{e}_{\rm w}=0.5$', r'$+\rho_{\star}=0.08$', r'$+\epsilon_{\rm r}=0.22$']
#    colors = ['green', 'red']
#    colors = ['blue', 'orange', 'mediumseagreen', 'magenta', 'cyan']
#    param_defaults = []
#    snapshots = [12]
#    snapshots = [8]
#    plot_name = "L302_observables_sfrd_jk"
#    plot_name = "L68_calibration"
#    show_spread = True
#    show_spread = False

#    file_ending = "all"   # use for simulation tests
#    file_ending = "nDGP"   # nDGP project
#    file_ending = "eff_data"   # fR hydro+SZ project
#    file_ending = "low_mass"   # for fR and nDGP pressure profiles

#    mass_cut = 4e12   # for fR hydro project, units Msun/h
#    mass_cut = 4.4e12   # for nDGP hydro project, units Msun/h
#    mass_cut = 1.4e12   # for fR and nDGP pressure profiles
#    delta = 500   # 500 or 200 or "all" (wrt. TRUE density field)

#    core_frac = 0.15

#    mass_cut = 1e10   # 1e13 Msun
#    delta = "all"   # 500 or 200 or "all"
    
#    rescaling = "true"   # effective or true
#    use_analytical = True   # use tanh formula to find ratio?
#    proxy_types = ["T", "SZ", "Yx", "Lx"]   # T or SZ or Yx or Lx
#    temp_weight = "mass"
#    no_core = True   # True: exclude r < core_frac * R500 core region
#    mass_bins = ["low", "middle"]
#    for (rid, r) in enumerate(realisations):
#        for snapshot in snapshots:
#           break
#            cp = ClusterProperties(simulation, models[rid], r, snapshot, system=system[rid], mass_cut=mass_cut, delta=delta, file_ending=file_ending, rescaling=rescaling, core_frac = core_frac)
#            cp.cluster_properties()
#            for proxy in proxy_types:
#                cp.proxy_scaling_relation(proxy_type=proxy, no_core=no_core, temp_weight=temp_weight, use_analytical=use_analytical)
#            for mass_bin in mass_bins:
#                cp.profile(mass_bin = mass_bin)

#    sr = SimulationTesting(simulation, models, realisations, snapshots[0], file_ending=file_ending, labels=plot_labels, colors=colors, defaults=param_defaults, plot_name=plot_name, system=system, show_spread=show_spread)
#   sr.tng_observables()
