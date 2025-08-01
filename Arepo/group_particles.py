#from mpl_toolkits import mplot3d
from numpy import *
from pylab import *
from matplotlib import *

import os
import pickle
from scipy.stats import binned_statistic

import sys
import imp
read_hdf5 = imp.load_source('read_hdf5', '/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/arepo_hdf5_library_old/read_hdf5.py')
search_grid = imp.load_source('search_grid', '/cosma8/data/dp203/bl267/Projects/Ongoing/Clusters/arepo_hdf5_library_old/search_grid.py')


class GroupParticles:
    '''This class finds the particles within a certain radius around each group and stores them in a file in the simulation directory.
    If the file is present, it is loaded. '''
    def __init__(self, s, outfile, snapnum, radius = 1., ngroups = -1, gridsize = 16, ptypes = ['gas', 'dm', 'stars', 'bh'], parttypes = [0,1,4,5], mass_cut=1e12, delta="all", hdf5=True):
        self.dumpfile = outfile + "group_particles_" + str(snapnum).zfill(3) + ".pickle"
#        print("Deleting group particle file")
#        os.remove(self.dumpfile)
#        print("File Removed!")
        self.s = s
        self.ptypes = ptypes
        self.parttypes = parttypes
        self.mass_cut = mass_cut
        self.delta = delta
        self.hdf5 = hdf5
        if len(parttypes) != len(ptypes):
            raise ValueError("wrong particle type lists")

        if os.path.exists(self.dumpfile):
            print("Group particle file %s exists!" % (self.dumpfile))
            f = open(self.dumpfile, "rb")
            (self.radius, self.groups, self.ptypes, self.group_particles) = pickle.load(f)
            f.close()
            if self.radius != radius:
                raise ValueError("stored radius", self.radius, " not equal to radius", radius)
            if len(self.groups) != ngroups and ngroups > 0:
                raise ValueError("wrong number of groups in file")
            
        else:
            print("Group particle file %s does not exist!" % (self.dumpfile))
            self.group_particles = {}
            self.radius = radius
            self.gridsize = gridsize
            self.ngroups = ngroups

            self.store_group_particles()


    def store_group_particles(self):
        print("Creating new group particle data")
        sg = search_grid.SearchGrid(self.s, ptypes = self.parttypes, gridsize = self.gridsize, hdf5 = self.hdf5)


        if self.ngroups > 0:
            self.groups = arange(self.ngroups)
        else:
            self.groups = arange(self.s.cat['Ngroups_Total'])

        for pt in self.ptypes:
            self.group_particles[pt] = empty(len(self.groups), dtype = 'object')


        for (g, group) in enumerate(self.groups):
            if self.delta == 500:
                if self.s.cat['Group_M_Crit500'][group] * self.s.header.hubble < self.mass_cut:
                    continue   # ignore clusters with M500 < mass_cut [Msun/h]
            elif self.delta == 200:
                if self.s.cat['Group_M_Crit200'][group] * self.s.header.hubble < self.mass_cut:
                    continue   # ignore clusters with M200 < mass_cut [Msun/h]
            # if delta is neither 200 or 500, all groups are used

            if g%500==0:
                print("Processing group ", group)
                print("Mass M500 [Msun/h]: ", self.s.cat['Group_M_Crit500'][group] * self.s.header.hubble)

            for (pi, pt) in enumerate(self.ptypes):
                part = sg.find_particles(self.s.cat['GroupPos'][group], self.radius*self.s.cat['Group_R_Crit200'][group], parttype = pt)
                self.group_particles[pt][g] = part

        
        print("Saving new group particle data..")    
        f = open(self.dumpfile, "wb")
        pickle.dump((self.radius, self.groups, self.ptypes, self.group_particles), f)
        f.close()
        print("Save was successful!")

        
