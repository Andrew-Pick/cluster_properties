"""
Routines for reading Gadget and AREPO snapshots and group catalogs in a convenient way
The library needs the readsnap and readsubf routines if Gadget formats are to be read.
In order to read data from a snapshot or group catalog, an instance of the snapshot class has to be created specifying simulation directory and snapshot number.
Author: Christian Arnold
Co-Author: Christoph Becker
"""
import numpy as np
import pickle
import os
import sys
import glob
import h5py

from readsnap import snapshot_header
from readsnap import read_block

def E(a, omega_m, omega_l, omega_r = 4.67e-5):
    return np.sqrt(omega_r * a**(-4) + omega_m * a**(-3) + omega_l)

def cosmic_time(a, omega_m, omega_l, h, omega_r = 4.67e-5):
    H0 = 100*h

    a_factors = np.linspace(0, a, 100000)
    da = a_factors[1:] - a_factors[:-1]
    a_factors = (a_factors[1:] + a_factors[:-1]) / 2.

    integral = sum(da / (E(a_factors, omega_m, omega_l, omega_r) * a_factors))
    cosmic_time = 1./H0 * integral *1e12
    return cosmic_time

class constants:
    """Class to easily access commonly used constants within a snapshot class"""

    def __init__(self, snapshot):
        '''
        Initialise the constants class for a given snapshot object. 

        Args:
            snapshot: a snapshot class object
        '''
        self.h = snapshot.header.hubble
        self.G = np.float64(
            6.67408e-11 / (3.08567758e22 ** 3) * 1.9891e30
        )  # Mpc**3/(M_solar*s**2) ## 6.67398e-11 m3 kg-1 s-2
        self.H = 100 * self.h / 3.08567758e22 * 1000  # 1/s
        self.Ht = np.sqrt(self.H**2 * (snapshot.header.omega_m / snapshot.header.time**3 + snapshot.header.omega_l))
        self.kB = 1.38064852e-16  # erg/K, Boltzmann const.
        self.rho_crit = np.float64(3 * (self.H ** 2) / (8 * np.pi * self.G))  # M_solar/Mpc**3
        self.rho200 = 200 * self.rho_crit  # M_solar/Mpc**3
        self.keV = np.float64(1000.0 * 1.602e-19)  # J/keV
        self.mproton = np.float64(1.67262e-27)  # kg
        self.f = 0.76  # [], Xh hydrogen fraction
        self.mmean = (1.0 + (1.0 - self.f) / self.f) / (
            2.0 + 3.0 * (1.0 - self.f) / (4.0 * self.f)
        )
        self.Mpc = 3.08567758e22  # m
        self.c = 3.0e8  # m/s
        self.c_Mpc = 3.0e8 / self.Mpc  # Mpc/s

class header:
    """Class containing the header part of a snapshot"""

    def __init__(self, snapshot):
        self.filename = snapshot.headername
        self.format = 3

        f = h5py.File(self.filename, "r")

        # simulation parameters physical
        self.attrs = list(f["/Header"].attrs.items())
        self.npart = f["/Header"].attrs["NumPart_ThisFile"]
        self.num_total = f["/Header"].attrs["NumPart_Total"].astype(np.int64)
        self.massarr = f["/Header"].attrs["MassTable"]
        self.time = f["/Header"].attrs["Time"]
        self.redshift = f["/Header"].attrs["Redshift"]
        self.sfr = f["/Header"].attrs["Flag_Sfr"]
        self.feedback = f["/Header"].attrs["Flag_Feedback"]
        self.nall = f["/Header"].attrs["NumPart_Total"]
        self.cooling = f["/Header"].attrs["Flag_Cooling"]
        self.filenum = f["/Header"].attrs["NumFilesPerSnapshot"]
        self.boxsize = f["/Header"].attrs["BoxSize"]
        self.omega_m = f["/Header"].attrs["Omega0"]
        self.omega_l = f["/Header"].attrs["OmegaLambda"]
        self.hubble = f["/Header"].attrs["HubbleParam"]
        self.swap = 0
        self.cosmic_time = cosmic_time(self.time, self.omega_m, self.omega_l, self.hubble)

        # simulation parameters units
        try:
            self.unitlength = f["/Header"].attrs["UnitLength_in_cm"]
            self.unitmass = f["/Header"].attrs["UnitMass_in_g"]
            self.unitvelocity = f["/Header"].attrs["UnitVelocity_in_cm_per_s"]
        except:
            print("No units in snapshot header, using 1.")
            self.unitlength = 1
            self.unitmass = 1
            self.unitvelocity = 1
            
        f.close()


class hdf5_names:
    """
    Class to translate the old four-letter identifiers to the hdf5 names
    in the snaopsnot and group files
    """

    def __init__(self):
        self.name = {}
        self.name["POS "] = "Coordinates"
        self.name["MASS"] = "Masses"
        self.name["VEL "] = "Velocities"
        self.name["ID  "] = "ParticleIDs"
        self.name["U   "] = "InternalEnergy"
        self.name["RHO "] = "Density"
        self.name["VOL "] = "Volume"
        self.name["NE  "] = "ElectronAbundance"
        self.name["NH  "] = "NeutralHydrogenAbundance"
        self.name["HSML"] = "SmoothingLength"
        self.name["SFR "] = "StarFormationRate"
        self.name["AGE "] = "StellarFormationTime"
        self.name["Z   "] = "Metallicity"
        self.name["BHMA"] = "BH_Mass"
        self.name["ACCE"] = "Acceleration"
        self.name["MGPH"] = "ModifiedGravityPhi"
        self.name["MGGP"] = "ModifiedGravityGradPhi"
        self.name["MGAC"] = "ModifiedGravityAcceleration"

    def translate_blocklist_to_gadget(self, blocklist):
        """Helper method to translate a blocklist to AREPO terminology."""
        if type(blocklist) == str:
            blocklist = [blocklist]


        for block in blocklist:
            if len(block) == 4:
                return blocklist

        
        new_blocklist = []
        for block in blocklist:
            new_blocklist.append(list(self.name.keys())[list(self.name.values()).index(block)])

        return new_blocklist

    def translate_blocklist_to_arepo(self, blocklist):
        """Helper method to translate a blocklist to old Gadget terminology."""
        if type(blocklist) == str:
            blocklist = [blocklist]

        translate = True
        for block in blocklist:
            if len(block) != 4 or block == "Mass":
                translate = False
                break

        if not translate:
            return blocklist

        else:
            new_blocklist = []
            for block in blocklist:
                new_blocklist.append(self.name[block])

        return new_blocklist
        

class snapshot:
    """
    Snapshot class; reads header and constants at
    initialisation; contains methods for reading particle data and group catalogs.
    Create an instance of the class using:
    my_snapshot = snapshot(snapnum, <directory>, <dirbases>, <snapbases>, <exts>)

    Usage Example:

    my_snapshot = snapshot(30, "/my/simulation/directory")

    This will load snapshot number 30 in the specified directory.
    """

    def __init__(
        self,
        snapnum:int,
        directory:str="./",
        dirbases:list=["snapdir_", ""],
        snapbases:list=["/snap_"],
        exts:list=[".0.hdf5", ".hdf5", "", ".0"],
        check_total_particle_number:bool=False,
        part_type_list:list = ['gas', 'dm', None, 'tracers', 'stars', 'bh'],
        verbose:bool = False
    ):

        '''
        Initialise the snapshot class object and read the snapshot header. If required,
        the total number of particles is calculated. 

        Args:
            snapnum     The simulation output number.
            <directory>     The output directory of the simualtion, optional, default './'.
            <dirbases>      A list of possible directory names for the snapshot directories, optional (normally not needed), default '["snapdir_", ""]'
            <snapbases>     A list of possible snapshot names, optional (normally not needed), default '["snap_"]'    
            <exts>      A list of possible file extensions, optional (normally not needed), default '["", ".hdf5"]'
            <check_total_partcile_number>   Wheteher to check the total partciles number in the simulation instead of relying on the header value (which can be wrong)
            <part_type_list>  Which particle types correspond to which particle type number, default ['gas', 'dm', None, 'tracers', 'stars', 'bh']
            <verbose>   Whether to generate lots of debugging output, default False
        '''
        self.directory = directory
        self.snapnum = snapnum
        self.check_total_particle_number = check_total_particle_number
        self.part_type_list = part_type_list
        self.names = hdf5_names()

        found_files = False
        for dirbase in dirbases:
            for snapbase in snapbases:
                for dirnum in [str(snapnum).zfill(3), ""]:
                    for ext in exts:
                        try_file = (
                            directory + dirbase + dirnum + snapbase + dirnum + ext
                        )
                        if verbose:
                            print("Trying file:", try_file)
                        if os.path.exists(try_file):
                            self.headername = try_file
                            self.snapname = (
                                directory + dirbase + dirnum + snapbase + dirnum
                            )
                            self.ext = ''
                            if len(ext) > 0:
                                if ext[-1] == '5':
                                    self.ext = '.hdf5'
                            found_files = True

        if not found_files:
            print("Headerfiles of %s not found." % directory)
            sys.exit()
        else:
            print("Headername: " + self.headername)
            print("Sanpname: " + self.snapname)

        # --- use new routine only for hdf5 snapshots ---
        if self.headername[-4:] == "hdf5":
            self.hdf5 = True
            self.header = header(self)
            
        # --- otherwise import readsnap and readsubf to read the old gadget format ---
        else:
            self.hdf5 = False
            self.header = snapshot_header(self.headername)

        self.time = self.header.time
        self.const = constants(self)
        self.data = {}

        # --- Calculate the total number of particles by hand, if required ---
        if self.check_total_particle_number:
            self.get_tot_num_part()

    def read(self, blocklist, parttype=-1, partition=[1, 0]):
        """Reading method to load particle data from snapshots.
        my_snapshot.read(blocklist, parttype = [0,1])

        Arguments:
        blocklist    List of hdf5 block names to be read (see: 'my_snapshot.show_snapshot_contents()')
        parttype     List of parttypes for which the data should be read, optional, default '-1' (read all types)

        Usage Example: 

        my_snapshot.read(['Velocities', 'Coordinates'], parttype = [0,1])

        Will read coordinates and velocities for gas and dm from the snapshot.
        The data is accessible through 

        my_snapshot.data
        """
        print("Reading " + str(blocklist) + "from snapshot")
        if type(blocklist) == str:
            blocklist = [blocklist]
        if not self.hdf5:  # use the old method to read gadget type snapshots
            self.read_gadget_format(blocklist, parttype)
        else:  # use the faster hdf5 reading routines
            self.read_hdf5(blocklist, parttype, partition)

    def read_gadget_format(self, blocklist, parttype = -1):
        '''
        Use the readsubf and readsnap routines to read old gagdet type snapshots.
        Called (automatically) by the 'read' routine if snapshot format is not hdf5.
        The loaded snapshot data is available in the same format as the hdf5 data.

        Args:
            blocklist: list of blocks to read (in AREPO terminology)
            parttype: either -1 (read all partcile type, default) or a list of particle types

        '''
        blocklist = self.names.translate_blocklist_to_arepo(blocklist)
        gadget_blocklist = self.names.translate_blocklist_to_gadget(blocklist)


        for b, block in enumerate(blocklist):
            gadget_block = gadget_blocklist[b]
            
            factor = self.get_unit_factor(block)
            self.data[block] = read_block(self.snapname, 
                                          gadget_block, 
                                          parttype=parttype) * factor


    def get_unit_factor(self, block):
        """Helper method"""
        self.length_blocks = [
            "GroupCM",
            "Coordinates",
            "GroupPos",
            "Group_R_Crit200",
            "Group_R_Vir_Eff",
            "Group_R_Crit500",
            "Group_R_Mean200",
            "Group_R_TopHat200",
            "SubhaloCM",
            "SubhaloHalfmassRad",
            "SubhaloHalfmassRadType",
            "SubhaloPos",
            "SubhaloVmaxRad",
        ]
        self.mass_blocks = [
            "Masses",
            "ModifiedGravityEffectiveMass",
            "SubhaloMass",
            "SubhaloMassInHalfRad",
            "SubhaloMassInHalfRadType",
            "SubhaloMassInMaxRad",
            "SubhaloMassInMaxRadType",
            "SubhaloMassInRad",
            "SubhaloMassInRadType",
            "SubhaloMassType",
            "Group_M_Crit200",
            "Group_M_Vir_Eff",
            "Group_M_In_R_Vir_Eff",
            "Group_M_Eff_In_R_Crit200", 
            "Group_M_Eff_In_R_Crit500",
            "Group_M_Crit500",
            "Group_M_Mean200",
            "Group_M_TopHat200",
            "Group_MassType_Crit200",
            "Group_MassType_Crit500",
            "Group_MassType_Mean200",
            "Group_MassType_TopHat200",
            "GroupMass",
            "GroupMassType",
        ]
        if block in self.length_blocks:
            factor = 1.0 / self.const.h
        elif block in self.mass_blocks:
            factor = 1e10 / self.const.h
        else:
            factor = 1.0

        return factor

    def check_for_blocks(self, fnames, blocklist, parttype):
        """
        Helper method
        Parameters
        ----------
        
        Returns
        -------
        """
        self.blockpresent = {}
        f = h5py.File(fnames[0], 'r')

        if parttype == -1:
            parttype = []
            for pt in range(10):
                if "PartType" + str(pt) in list(f.keys()):
                    parttype.append(pt)

        for block in blocklist:

            self.blockpresent[block] = []
            for pt in parttype:
                if "PartType" + str(pt) in list(f.keys()):
                    if block in list(f["PartType" + str(pt) + "/"].keys()):
                        self.blockpresent[block].append(pt)
                    elif block == "Masses" and f["Header/"].attrs["MassTable"][pt] > 0:
                        self.blockpresent[block].append(-pt)

        print("Found %s data in %s" % (str(self.blockpresent), f.filename))

    def get_tot_num_part(self):
        """helper method"""
        self.header.num_total = np.zeros(6, dtype=np.int64)
        files = self.determine_files(self.snapname + ".")

        for fn in files:
            fname = self.snapname + "." + str(fn) + self.ext

            if self.hdf5:
                self.f = h5py.File(fname, 'r')
                part_this_file = self.f["/Header/"].attrs["NumPart_ThisFile"]
            else:
                h = rs.snapshot_header(fname)
                part_this_file = h.npart
            self.header.num_total += part_this_file
        print("Total number of particles:", self.header.num_total)

    def create_data_array(self, fnames, blocklist, partition=[1, 0]):
        """
        Helper method.
        
        Parameters
        ----------
        f : np.list
            List of file-names of a given snapshot
        blocklist : np.list
            List of physical quantities to output
        partition : np.list
            [# of partitions, which partition to access]
        """

        for block in blocklist:
            self.data[block] = {}
            for pt in self.blockpresent[block]:

                if len(fnames) == 1:
                    f = h5py.File(fnames[0], 'r')
                    datalen = self.header.num_total[pt]
                else:
                    datalen = 0
                    for fn in fnames:
                        f = h5py.File(fn, 'r')
                        datalen += f["/Header/"].attrs["NumPart_ThisFile"][pt]
                        if fn != fnames[-1]:
                            f.close()

                if pt >= 0:
                    datashape = 1
                    if len(f["PartType" + str(pt) + "/" + block + "/"].shape) > 1:
                        datashape = f["PartType" + str(pt) + "/" + block + "/"].shape[1]
                    datatype = f["PartType" + str(pt) + "/" + block + "/"].dtype
                    
                    if block == 'IntegerCoordinates':
                        datatype = np.float64

                    if datalen < self.header.npart[pt]:
                        raise ValueError(
                            "There are less rows in .hdf5 file than "
                            + "the header thinks there are! \n"
                            "Try setting check_total_particle_number"
                        )

                    if datashape > 1:
                        self.data[block][self.get_part_type_name(pt)] = np.zeros(
                            (datalen, datashape), dtype=datatype
                        )
                    else:
                        self.data[block][self.get_part_type_name(pt)] = np.zeros(
                            datalen, dtype=datatype
                        )
                else:
                    factor = self.get_unit_factor(block)
                    self.data[block][self.get_part_type_name(-pt)] = (
                        np.ones(self.header.num_total[-pt])
                        * f["Header/"].attrs["MassTable"][-pt]
                        * factor
                    )

    def show_snapshot_contents(self):
        """This function prints the available data fields contined in this snapshot.
        Usage:
        my_snapshot.show_snapshot_contents()
        """
        fname = self.snapname + "." + str(0) + ".hdf5"
        self.f = h5py.File(fname, 'r')

        print("")
        print("----------------------------------------")
        print("Snapshot data for file: " + fname)
        print("----------------------------------------")
        print("Available data fields:")
        print("----------------------------------------")
        for k in self.f.keys():
            print(k)

        for k in self.f.keys():
            if k in ["Header", "Config", "Parameters"]:
                print("----------------------------------------")
                print(k + " contents: ")
                print("----------------------------------------")
                for i in self.f[k].attrs.keys():
                    print(i, self.f[k].attrs[i])
                print("")
            else:
                print("")
                print("----------------------------------------")
                print("Contents of data field: " + k)
                print("----------------------------------------")
                for i in self.f[k].keys():
                    print(i)
        self.f.close()


    def read_hdf5(self, blocklist, parttype, partition=[1, 0]):
        """
        helper method
        Parameters
        ----------
        blocklist : str
            Indicates which physical quantity to load
        parttype : int
            [DM, Stars, Baryons, BH]
        partition : np.list
            Which files of a snapshot should be loaded
        """
        # list file id's
        files = self.determine_files(self.snapname + ".", partition)
        blocklist = self.names.translate_blocklist_to_arepo(blocklist)

        # prepare read_hdf5 output
        file_name = self.snapname + ".%d.hdf5"
        self.check_for_blocks([file_name % files[0]], blocklist, parttype)
        if partition[0] == 1:
            self.create_data_array([file_name % files[0]], blocklist)
        if partition[0] > 1:
            file_names = [file_name % fn for fn in files]
            self.create_data_array(file_names, blocklist, partition)

        fn = 0
        fname = file_name % fn
        f = h5py.File(fname, 'r')
        self.particle_types = len(f["/Header/"].attrs["NumPart_ThisFile"])
        f.close()

        self.partcounter = np.zeros(self.particle_types, dtype=np.int64)
        for fn in files:
            fname = file_name % fn
            if fn % 1 == 0:  # debug
                print("Reading file %s" % fname)
            f = h5py.File(fname, 'r')

            for block in blocklist:
                factor = self.get_unit_factor(block)
                name = block

                for pt in self.blockpresent[block]:
                    if pt >= 0:
                        if block == "IntegerCoordinates":
                            if f["/Header/"].attrs["NumPart_ThisFile"][pt] > 0:
                                self.data[block][self.get_part_type_name(pt)][
                                    self.partcounter[pt] : self.partcounter[pt]
                                    + f["/Header/"].attrs["NumPart_ThisFile"][pt]
                                ] = (
                                    f[
                                        "PartType" + str(pt) + "/" + block + "/"
                                    ][()].astype(np.float64)
                                    * factor / 2**32 * self.header.boxsize / self.const.h
                            )
                        else:
                            if f["/Header/"].attrs["NumPart_ThisFile"][pt] > 0:
                                self.data[block][self.get_part_type_name(pt)][
                                    self.partcounter[pt] : self.partcounter[pt]
                                    + f["/Header/"].attrs["NumPart_ThisFile"][pt]
                                ] = (
                                    f["PartType" + str(pt) + "/" + block + "/"][()]
                                    * factor
                                )

            self.partcounter += f["/Header/"].attrs["NumPart_ThisFile"]
            f.close()

    def get_part_type_name(self, type_id:int):
        '''
        Helper method to find a parttype name given its parttype ID.
        
        Args:
            type_id: the ID of the particle type
        Returns:
            The name of the corresponding particle type
        '''
        if type_id < len(self.part_type_list):
            return self.part_type_list[type_id]
        else:
            raise ValueError("Type_id", type_id, "part_type_list", self.part_type_list)
        

    def get_part_type_ID(self, part_type_name:str):
        '''
        Helper method to find a part type ID given a part type name.
        
        Args:
            part_type_name: the name of the part type to find.

        Returns:
            The ID of the part type in question.
        '''
        
        if part_type_name in self.part_type_list:
            return np.where(np.array(self.part_type_list) == part_type_name)[0][0]
        else:
            raise ValueError("Name", part_type_name, " not in part_type_list", self.part_type_list)


    def group_catalog(
        self,
        hdf5_names=["GroupPos", "Group_M_Crit200", "Group_R_Crit200"],
        masstab=True,
        group_veldisp=True,
        file_prefix="",
        files=-1,
        path="",
        dirname="groups_",
        filename="fof_subhalo_tab_",
    ):
        """Read data from the group catalog corresponding to the snapshot.
        Usage:
        my_snapshot.group_catalog(<hdf5_names>, <masstab>, <group_veldisp>, <file_prefix>, <files>, <path>, <dirname>, <filename>)

        Arguments:
        hdf5_names       List of hdf5 names of the data fields to be loaded (see  'my_snapshot.show_group_catalog_contents()'), optional, default '['GroupPos', 'Group_M_Crit200', 'Group_R_Crit200']'
        masstab      Only needed for Gadget format, optional
        group_veldisp    Only needed for Gadget format, optional 
        file_prefix      Prefix for the group directory, optional, default ''
        files        List of files to be loaded from the group catalog, optional, default '-1' (all files)
        path         path where the group catalog is stored, optional, default: same path as snapshot data
        dirname      directory name for the group catalog subdirectories, optional, default 'groups_'
        filename     filename for the individual catalog files, optional, default '/fof_subhalo_tab_'

        Example:
        my_snapshot.group_catalog(['GroupPos', 'SubhaloPos']) 
        This will load the positions of all groups and subhalos.
        """
        if not self.hdf5:
            self.cat = readsubf.subfind_catalog(
                self.directory + file_prefix,
                self.snapnum,
                masstab=masstab,
                group_veldisp=group_veldisp,
            )
        else:
            self.fast_group_catalog(
                hdf5_names=hdf5_names,
                files=files,
                path=path,
                dirname=dirname,
                filename=filename,
                file_prefix=file_prefix,
            )

    def show_group_catalog_contents(
        self, path="", dirname="groups_", filename="fof_subhalo_tab_", file_prefix=""
    ):
        """
        This Function will print the available data fields for the group catalog
        corresponding to this snapshot. 
        
        Usage:
        my_snapshot.show_group_catalog_contents()

        See 'group_catalog()' for optional arguments.

        """
        if path == "":
            path = (
                self.directory
                + file_prefix
                + "/"
                + dirname
                + str(self.snapnum).zfill(3)
                + "/"
                + filename
                + str(self.snapnum).zfill(3)
                + "."
            )

        fname = path + str(0) + ".hdf5"
        self.f = h5py.File(fname, 'r')
        print("")
        print("----------------------------------------")
        print("Group catalog data for file: " + fname)
        print("----------------------------------------")
        print("Header contents: ")
        print("----------------------------------------")
        for k in self.f["Header"].attrs.keys():
            print(k)
        print("")
        print("----------------------------------------")
        print("Group data: ")
        print("----------------------------------------")
        for k in self.f["Group/"].keys():
            print(k)
        print("")
        print("----------------------------------------")
        print("Subhalo data: ")
        print("----------------------------------------")
        for k in self.f["Subhalo/"].keys():
            print(k)
        print("----------------------------------------")
        self.f.close()

    def fast_group_catalog(
        self,
        hdf5_names=["GroupPos", "Group_M_Crit200", "Group_R_Crit200"],
        files=-1,
        path="",
        dirname="groups_",
        filename="fof_subhalo_tab_",
        file_prefix="",
        show_data=False,
    ):
        """Helper method"""
        if path == "":
            path = (
                self.directory
                + file_prefix
                + "/"
                + dirname
                + str(self.snapnum).zfill(3)
                + "/"
                + filename
                + str(self.snapnum).zfill(3)
                + "."
            )

        print("Reading" + str(hdf5_names) + "from hdf5 group catalog" + path)

        # --- find the files ---
        if files == -1:
            files = self.determine_files(path)

        self.cat = {}

        # --- iterate over files ---
        group_counter = 0
        sub_counter = 0
        for fn in files:
            fname = path + str(fn) + ".hdf5"
            self.f = h5py.File(fname, 'r')

            if fn % 10 == 0:
                print("Reading file " + fname)

            ng = self.f["Header/"].attrs["Ngroups_ThisFile"]
            ns = self.f["Header/"].attrs["Nsubgroups_ThisFile"]

            # --- create empty arrays for the data ---
            if fn == 0:
                # --- read header of the first file ---
                self.cat["n_groups"] = self.f["Header/"].attrs["Ngroups_Total"]
                self.cat["n_subgroups"] = self.f["Header/"].attrs["Nsubgroups_Total"]
                for key in self.f["/Header"].attrs.keys():
                    self.cat[key] = self.f["/Header"].attrs[key]

                # --- create data arrasys for groups and subhalos ---
                for hn in hdf5_names:
                    sh = 1

                    if hn[0] == "G":
                        if self.cat["n_groups"] == 0:
                            continue
                        if len(self.f["Group/" + hn][()].shape) > 1:
                            sh = self.f["Group/" + hn][()].shape[1]
                        if sh > 1:
                            self.cat[hn] = np.zeros((self.cat["n_groups"], sh))
                        else:
                            self.cat[hn] = np.zeros(self.cat["n_groups"])

                    elif hn[0] == "S":
                        if self.cat["n_subgroups"] == 0:
                            continue
                        if len(self.f["Subhalo/" + hn][()].shape) > 1:
                            sh = self.f["Subhalo/" + hn][()].shape[1]
                        if sh > 1:
                            self.cat[hn] = np.zeros((self.cat["n_subgroups"], sh))
                        else:
                            self.cat[hn] = np.zeros(self.cat["n_subgroups"])

                    else:
                        raise ValueError("can't deal with that", hn, hn[0])

            # --- read the data ---
            for hn in hdf5_names:
                unit_factor = self.get_unit_factor(hn)
                if hn[0] == "G" and ng > 0:
                    self.cat[hn][group_counter : group_counter + ng] = (
                        self.f["Group/" + hn][()] * unit_factor
                    )
                elif hn[0] == "S" and ns > 0:
                    self.cat[hn][sub_counter : sub_counter + ns] = (
                        self.f["Subhalo/" + hn][()] * unit_factor
                    )

            group_counter += ng
            sub_counter += ns
            self.f.close()

    def determine_files(self, path, partition=[1, 0]):
        """Helper Routine to count # of files"""
        if not os.path.exists(path + "0" + self.ext):
            raise ValueError("File", path + "0" + self.ext, " not found")

        file_names = path + "*"
        file_numbers = np.arange(len(glob.glob(file_names)))
        if partition[0] == 1:
            return file_numbers
        elif partition[0] > 1:
            return np.array_split(file_numbers, partition[0])[partition[1]]
        else:
            "Illegal partition of snapshot files!"


