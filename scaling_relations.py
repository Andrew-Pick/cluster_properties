#from __future__ import division
import sys
sys.path.insert(0, '/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/arepo_hdf5_library_old')
import group_particles
import read_hdf5
import numpy as np
import scipy.optimize as op
import scipy.stats as stat

import matplotlib
import matplotlib.gridspec as gridspec
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
        self.fileroot = "/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/%s/%s/" % (self.simulation,self.model)
        self.fileroot2 = "/cosma8/data/dp203/dc-pick1/cluster_properties/plots/"
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
        
        if self.model == "GR" or "F50":
            print("Skipping GR and F50 model data loading.")
            return
        
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


    def cluster_properties(self, group_id = -1):
        if self.model == "GR" or "F50" :
            print("Skipping GR and F50 model processing.")
            return
        
        
        if self.model == "GR" or self.simulation == "L302_N1136":
            group_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s%s.pickle" % (self.simulation, self.model, self.realisation,self.snapshot, self.file_ending, self.core_label)
            subhalo_dumpfile = self.fileroot+"pickle_files/subhalo_%s_%s_%s_s%d_%s.pickle" % (self.simulation,self.model,self.realisation,self.snapshot,self.file_ending)
            print(group_dumpfile)
            print(subhalo_dumpfile)
        else:
            group_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s_rescaling%s%s.pickle" % (self.simulation, self.model, self.realisation,self.snapshot, self.file_ending, self.rescaling, self.core_label)
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
            group_dumpfile_directory = os.path.dirname(group_dumpfile)
            os.makedirs(group_dumpfile_directory, exist_ok=True)
            print("Created directory:", group_dumpfile_directory)   


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
        if self.model[0] == "F" and self.snapshot in [40, 45, 50, 55, 60]:
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

        df = open(profile_dumpfile,"w+")
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
        if self.model=="F5" or self.model=="F50":
            f_R0 = 1e-5
        elif self.model=="F6" or self.model=="F60":
            f_R0 = 1e-6
        elif self.model=="F55":
            f_R0 = 10**(-5.5)
        elif self.model=="F45":                                                                                                                                                         
            f_R0 = 10**(-4.5)
        elif self.model=="F4" or self.model=="F40":
            f_R0 = 10**(-4)
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




class Scaling_Relation:
    def __init__(self, simulation, models, realisations, snapshot, file_ending, labels, colors, defaults, plot_name, system='cosma7', show_spread=False, property="SZ"):
        self.simulation = simulation
        self.models = models
        self.realisations = realisations
        self.snapshot = snapshot
        self.file_ending = file_ending
        self.labels = labels
        self.colors = colors
        self.defaults = defaults
        self.plot_name = plot_name
        self.system = system
        self.show_spread = show_spread
        self.property = property  # Property can be "SZ", "T", "Yx" or "Lx"
        if self.snapshot == 12:
            self.rshift = 0.5
        elif self.snapshot == 21:
            self.rshift = 0.0

        # load snapshot
#       if simulation == "L302_N1136":
#           if system[0] == "cosma7":
#               fd = FindData(simulation,models[0],"Wind_1",snapshot,system=system[0])
#
#       self.particle_path = fd.particle_path
#       self.group_path = fd.group_path
#       self.s = read_hdf5.snapshot(snapshot, directory=self.particle_path, dirbases=["snapdir_", ""], snapbases=["/GR_", "/gadget", "/snap_"], check_total_particle_number=True)

        self.fileroot = "/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/%s/" % (simulation)
        self.fileroot2 = "/cosma/home/dp203/dc-pick1/cluster_properties/plots/%s/" % (simulation)
        self.dumpfiles = [self.fileroot+"%s/pickle_files/%s_%s_%s_s%d_%s.pickle" % (m,simulation,m,realisations[mid],snapshot,file_ending) for (mid, m) in enumerate(models)]
        self.subhalo_dumpfiles = [self.fileroot+"%s/pickle_files/subhalo_%s_%s_%s_s%d_%s.pickle" % (m,simulation,m,realisations[mid],snapshot,file_ending) for (mid, m) in enumerate(models)]

    def redshift(self):
        self.mysize = "large"
        self.axsize = "medium"
        self.legsize = "small"
        self.ms = 5.
        self.lw = 1.
        fig = plt.figure(figsize=(4.2, 6)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0)  

        ax_main = plt.subplot(gs[0])
        ax_sub = plt.subplot(gs[1], sharex=ax_main)

        self.z0(ax_main)

        self.z0_subplot(ax_sub)

        # Hide x-ticks for ax_main and set the labels for ax_main
        plt.setp(ax_main.get_xticklabels(), visible=False)

        
        ax_main.set_ylabel(r'$\log_{10}(\overline{T}_{\textnormal{gas}} \, [\textnormal{keV}])$', fontsize=self.mysize, labelpad=10)
        ax_sub.set_ylabel(r'$\Delta \overline{T}_{\textnormal{gas}} / \overline{T}_{\textnormal{gas,GR}}$', fontsize=self.axsize, labelpad=15)
        ax_sub.set_xlabel(r'$\log_{10}(M_{500} \ [M_{\odot}])$', fontsize=self.mysize)
             
    
        plt.tight_layout()       
        directory = os.path.dirname(self.fileroot2)
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(self.fileroot2+"L302_gastempscalingz0_sub_mass_compare_rescaled.pdf")
        print('Figure saved at '+self.fileroot2+"L302_gastempscalingz0_sub_mass_compare_rescaled.pdf")

    def correction(self, truemass, model, redshift):
        '''Enhancement of the dynamical mass with respect to the true mass,
        given the true mass in units Msun, for z=0.
        '''
        
        p1=2.21
        f_R0 = None
        if model=="F5" or model=="F50":
            f_R0 = 1e-5
        elif model=="F6" or model=="F60":
            f_R0 = 1e-6
        elif model=="F55":
            f_R0 = 10**(-5.5)
        elif model=="F45":                                                                                                                                             
            f_R0 = 10**(-4.5)
        elif model=="F4" or model=="F40":
            f_R0 = 10**(-4)
        
        if f_R0 is None:
            raise ValueError(f"f_R0 not set for model {model}")

        fR = f_R(f_R0, 1 / (1. + redshift), 0.3089, 0.6911)
        p2 = 1.503 * np.log10(fR / (1. + redshift)) + 21.64  # 10^p2 units Msun/0.697
            #   p2_rescaled = np.log10(10**p2 / 0.697)   # 10^p2' units Msun
        return 7 / 6 - (1 / 6) * np.tanh(p1 * (np.log10(truemass) - p2))

    def z0(self, ax):
        
        self.median_prop_gr = None
        self.mean_log_mass_gr = None
        for (mid,m) in enumerate(self.models):
            ld = LoadDumpfile(self.dumpfiles[mid], self.subhalo_dumpfiles[mid])
            ldmass = ld.M500
            mass = ldmass # /0.6774
            if self.property == "T":
                prop = ld.mass_T500_with_core
                prop1 = ld.mass_T500  # prop1 refers to properties excluding the core region of clusters
            elif self.property == "SZ":
                prop = ld.Ysz_with_core
                prop1 = ld.Ysz_no_core
            elif self.property == "Yx":
                prop = ld.Mg500 * ld.mass_T500_with_core
                prop1 = ld.Mg500 * ld.mass_T500
            elif self.property == "Lx":
                prop = ld.Lx_with_core
                prop1 = ld.Lx_no_core
            prop_rescaled_list=[]
            prop_rescaled_list1=[]
            for i, p in zip(mass, prop):
                if m!= 'GR':
                    massratio = self.correction(truemass=i, model=m, redshift=self.rshift)
                    p_rescaled = p / massratio
                else:
                    p_rescaled = p
                prop_rescaled_list.append(p_rescaled)
            prop_rescaled = np.array(prop_rescaled_list)   
            
            for i, p in zip(mass, prop1):
                if m!= 'GR':
                    massratio = self.correction(truemass=i, model=m, redshift=self.rshift)
                    p_rescaled1 = p / massratio
                else:
                    p_rescaled1 = p
                prop_rescaled_list1.append(p_rescaled1)
            prop_rescaled1 = np.array(prop_rescaled_list1)   
            
            bins = np.logspace(np.log10(1.e13), 15.4, 9, base=10.0)
            digitized = np.digitize(mass, bins)
            logmass=np.log10(mass)
            logprop=np.log10(prop)
            logprop_rescaled=np.log10(prop_rescaled)
            logprop1=np.log10(prop1)
            logprop1_rescaled=np.log10(prop_rescaled1)
      
            
            mean_log_mass = np.array([(float(np.mean(np.log10(mass[digitized == i])))) for i in range(1, len(bins))])   # units Msun
            median_prop = np.array([float(np.median(np.log10(prop[digitized == i]))) for i in range(1, len(bins))])
            median_prop_no_log = np.array([float(np.median(prop[digitized == i])) for i in range(1, len(bins))])
            median_prop_rescaled = np.array([float(np.median(np.log10(prop_rescaled[digitized == i]))) for i in range(1, len(bins))]) 
            
            median_prop1 = np.array([float(np.median(np.log10(prop1[digitized == i]))) for i in range(1, len(bins))])
            median_prop_no_log1 = np.array([float(np.median(prop1[digitized == i])) for i in range(1, len(bins))])
            median_prop_rescaled1 = np.array([float(np.median(np.log10(prop_rescaled1[digitized == i]))) for i in range(1, len(bins))]) 

            size = np.array([len(prop[digitized == i]) for i in range(1, len(bins))])
            self.size_gr = np.array([len(prop[digitized == i]) for i in range(1, len(bins))])       
            size_mask = size >= 2
            mean_log_mass_main = mean_log_mass[size_mask]
            median_prop_main1 = median_prop1[size_mask]
            median_prop_rescaled_main1 = median_prop_rescaled1[size_mask]

            # handle sparse high-mass bins separately
            if np.any(~size_mask):  # there are sparse bins
                high_mass_mask = ~size_mask
                combined_mass = np.mean([np.mean(np.log10(mass[digitized == i])) for i in range(1, len(bins)) if high_mass_mask[i-1]])
                combined_prop1 = np.mean([np.median(np.log10(prop1[digitized == i])) for i in range(1, len(bins)) if high_mass_mask[i-1]])
                combined_prop_rescaled1 = np.mean([np.median(np.log10(prop_rescaled1[digitized == i])) for i in range(1, len(bins)) if high_mass_mask[i-1]])

            	# append as a single extra high-mass point
                mean_log_mass_main = np.append(mean_log_mass_main, combined_mass)
                median_prop_main1 = np.append(median_prop_main1, combined_prop1)
                median_prop_rescaled_main1 = np.append(median_prop_rescaled_main1, combined_prop_rescaled1)

            if m == "GR":
                self.median_prop_gr = median_prop_no_log
                self.mean_log_mass_gr = mean_log_mass
#                ax.scatter(logmass,logprop, marker='o', s=0.8, color="darkgrey",alpha=0.4)
#                ax.plot(mean_log_mass[size >= 5], median_prop[size >= 5], linewidth=self.lw, color=self.colors[mid],label = 'GR with core')  
                self.median_prop_gr1 = median_prop_no_log1
                self.mean_log_mass_gr = mean_log_mass
                ax.scatter(logmass, logprop1, marker='o', s=0.8, color="darkgrey",alpha=0.8)
                ax.plot(mean_log_mass[size >= 2], median_prop1[size >= 2], linewidth=self.lw, color=self.colors[mid],label = 'GR no core')  
            else:
#                ax.plot(mean_log_mass[size >= 5], median_prop[size >= 5], linewidth=self.lw, linestyle='dotted', color=self.colors[mid],label =m+' with core',alpha=0.5)  
#                ax.plot(mean_log_mass[size >= 5], median_prop_rescaled[size >= 5], linewidth=self.lw, color=self.colors[mid],label = m+' with core rescaled',alpha=0.5) 
                ax.plot(mean_log_mass[size >= 2], median_prop1[size >= 2], linewidth=self.lw, linestyle='dotted', color=self.colors[mid],label =m+' no core')  
                ax.plot(mean_log_mass[size >= 2], median_prop_rescaled1[size >= 2], linewidth=self.lw, linestyle='dashed',  color=self.colors[mid],label = m+' no core rescaled') 
             
    
        ax.set_xlim([13, 15.4])
        ax.set_ylim([-0.75, 1.25])
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_yticklabels(r'')
        ax.set_yticks(np.arange(-0.75,1.25,0.25))
        ax.set_xticklabels(r'')
        ax.set_xticks(np.arange(13,15.4,0.5))


        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc='lower right', fontsize=self.legsize, frameon=False)

#        ax.set_xlabel(r'$\log_{10}(M_{500} \ [M_{\odot}])$', fontsize=self.mysize)
#        ax.set_ylabel('$\log_{10}(\overline{T}_{gas} \, [keV])$', fontsize=self.mysize)

        ax.set_yticklabels([r'',r'$-0.5$',r'',r'$0.0$',r'',r'$0.5$',r'',r'$1.0$'],fontsize=self.axsize)  

        offset_x_fraction = 0.05  # Horizontal offset
        offset_y_fraction = 0.05  # Vertical offset

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

# Apply the offset to the axis limits
        title_x = xlim[0] + (xlim[1] - xlim[0]) * offset_x_fraction
        title_y = ylim[1] - (ylim[1] - ylim[0]) * offset_y_fraction

        ax.text(title_x,title_y,rf'$\mathit{{z}} = {self.rshift:.1f}$',fontsize=self.mysize,ha='left', va='top')
 #       plt.tight_layout()
    
    
    def z0_subplot(self, ax):
        if self.median_prop_gr is None or self.size_gr is None:
            raise ValueError("GR data has not been processed yet.")

        for (mid, m) in enumerate(self.models):
            if m != "GR":  # Skip the GR model itself
                ld = LoadDumpfile(self.dumpfiles[mid], self.subhalo_dumpfiles[mid])
                ldmass = ld.M500
                mass = ldmass # /0.6774
                if self.property == "T":
                    prop = ld.mass_T500_with_core
                    prop1 = ld.mass_T500  # prop1 refers to properties excluding the core region of clusters
                elif self.property == "SZ":
                    prop = ld.Ysz_with_core
                    prop1 = ld.Ysz_no_core
                elif self.property == "Yx":
                    prop = ld.Mg500 * ld.mass_T500_with_core
                    prop1 = ld.Mg500 * ld.mass_T500
                elif self.property == "Lx":
                    prop = ld.Lx_with_core
                    prop1 = ld.Lx_no_core
                prop_rescaled_list = []
                prop_rescaled_list1 = []
                for i, p in zip(mass, prop):
                    massratio = self.correction(truemass=i, model=m, redshift=self.rshift)
                    p_rescaled = p / massratio
                    prop_rescaled_list.append(p_rescaled)
                prop_rescaled = np.array(prop_rescaled_list) 
                for i, p in zip(mass, prop1):
                    massratio = self.correction(truemass=i, model=m, redshift=self.rshift)
                    p_rescaled1 = p / massratio
                    prop_rescaled_list1.append(p_rescaled1)
                prop_rescaled1 = np.array(prop_rescaled_list1) 
                
                
                bins = np.logspace(np.log10(1.e13), 15.4, 9, base=10.0)
                digitized = np.digitize(mass, bins)
                size = np.array([len(mass[digitized == i]) for i in range(1, len(bins))])
                size_mask = self.size_gr >= 3
                # Calculate the ratio of the difference
                median_prop_rescaled = np.array([float(np.median(prop_rescaled[digitized == i])) for i in range(1, len(bins))if size_mask[i-1]])
                median_prop = np.array([float(np.median(prop[digitized == i])) for i in range(1, len(bins))if size_mask[i-1]]) 
                median_prop_rescaled1 = np.array([float(np.median(prop_rescaled1[digitized == i])) for i in range(1, len(bins))if size_mask[i-1]])
                median_prop1 = np.array([float(np.median(prop1[digitized == i])) for i in range(1, len(bins))if size_mask[i-1]]) 
                mean_log_mass = np.array([float(np.mean(np.log10(mass[digitized == i]))) for i in range(1, len(bins)) if size_mask[i-1]])
                median_prop_gr_filtered = self.median_temp_gr[size_mask]
                median_prop_gr_filtered1 = self.median_temp_gr1[size_mask]

                if len(median_prop_rescaled) != len(median_prop_gr_filtered):
                    raise ValueError(f"Array length mismatch: {len(median_prop_rescaled)} vs {len(median_prop_gr_filtered)}")
 
                ratio_diff_rescaled = (median_prop_rescaled - median_prop_gr_filtered) / median_prop_gr_filtered
                ratio_diff = (median_prop - median_prop_gr_filtered) / median_prop_gr_filtered
                ratio_diff_rescaled1 = (median_prop_rescaled1 - median_prop_gr_filtered1) / median_prop_gr_filtered1
                ratio_diff1 = (median_prop1 - median_prop_gr_filtered1) / median_prop_gr_filtered1
                # Plot the ratio differences
#               ax.plot(mean_log_mass, ratio_diff_rescaled,linewidth=self.lw,  color=self.colors[mid],alpha=0.5)
#               ax.plot(mean_log_mass, ratio_diff, linewidth=self.lw, linestyle='dotted', color=self.colors[mid],alpha=0.5)  
                ax.plot(mean_log_mass, ratio_diff_rescaled1,linewidth=self.lw, linestyle='dashed', color=self.colors[mid])
                ax.plot(mean_log_mass, ratio_diff1, linewidth=self.lw, linestyle='dotted', color=self.colors[mid])  
                
        ax.set_xlim([13, 15.4])
        ax.set_ylim([-0.2, 0.8])                                                                                                                                                               
        ax.tick_params(direction='in', width=1, top=True, right=True, which='both')
        ax.set_yticklabels(r'')
        ax.set_yticks(np.arange(-0.2,0.8,0.2))
        ax.set_xticklabels(r'')
        ax.set_xticks(np.arange(13,15.4,1))
  
        
        ax.xaxis.set_tick_params(width=1.5)
        ax.yaxis.set_tick_params(width=1.5)
        ax.legend(loc='lower right', fontsize=self.legsize, frameon=False)

        ax.set_xticklabels([r'$13$',r'$14$',r'$15$'],fontsize=self.axsize)
        ax.set_yticklabels([r'',r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$'],fontsize=self.axsize)  
        
        ax.set_ylabel(r'$\Delta T_{\mathrm{gas}} / T_{\mathrm{gas,GR}}$', fontsize=self.axsize)
        ax.set_xlabel(r'$\log_{10}(M_{500} \ [M_{\odot}])$', fontsize=self.mysize)


if __name__ == "__main__":
    simulation = "L302_N1136"
    models = ["GR","F60","F55","F50","F45","F40"]
    realisations = ["1", "1","1","1","1","1"]
    system = ["cosma8", "cosma6","cosma6","cosma6","cosma6","cosma6"]
    plot_labels = [r'L302-N1136-GR', r'L302-N1136-F60',r'L302-N1136-F55',r'L302-N1136-F50',r'L302-N1136-F45',r'L302-N1136-F40']
    colors = ['black', 'blue','magenta','green','orange','red']
    param_defaults = []
    snapshots = [21]
    plot_name = "L302_scaling_Relations"
    show_spread = True

    file_ending = "all"   # use for simulation tests

    core_frac = 0.15

    mass_cut = 5e13   # 1e13 Msun
    delta = "all"   # 500 or 200 or "all"
    
    rescaling = "true"   # effective or true
    use_analytical = True   # use tanh formula to find ratio?
    proxy_types = ["T"]   # T or SZ or Yx or Lx
    temp_weight = "mass"
    no_core = True   # True: exclude r < core_frac * R500 core region
    mass_bins = ["low", "middle"]
#   for (rid, r) in enumerate(realisations):
#       for snapshot in snapshots:
#           print(snapshot)
#           cp = ClusterProperties(simulation, models[rid], r, snapshot, system=system[rid], mass_cut=mass_cut, delta=delta, file_ending=file_ending, rescaling=rescaling, core_frac = core_frac)
#           cp.cluster_properties()
#            for proxy in proxy_types:
#                cp.proxy_scaling_relation(proxy_type=proxy, no_core=no_core, temp_weight=temp_weight, use_analytical=use_analytical)
#            for mass_bin in mass_bins:
#                cp.profile(mass_bin = mass_bin)
    print('data accessed')
    sr = Scaling_Relation(simulation, models, realisations, snapshots[0], file_ending=file_ending, labels=plot_labels, colors=colors, defaults=param_defaults, plot_name=plot_name, system=system, show_spread=show_spread)
    sr.redshift()
