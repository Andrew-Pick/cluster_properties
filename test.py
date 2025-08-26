import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import h5py
import pickle
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

np.random.seed(1273)

# --- constants in cgs ---
sigma_T = 6.6524587158e-25  # cm^2
m_e_c2  = 8.18710565e-7     # erg
# (If your P_e is in erg/cm^3 and dl in cm, prefactor is sigma_T / m_e_c2)

# --- cosmology helpers (you can use astropy.cosmology instead) ---
def comoving_distance(z):
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    d_comoving = cosmo.comoving_distance(z).value  # in Mpc
    print(f"Comoving distance at z={z} is {d_comoving:.2f}")
    return d_comoving*0.6774

def angular_diameter_distance(z):
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)  # matches your simulation
    d_ang = cosmo.angular_diameter_distance(z).value     # in Mpc
    print(f"Angular diameter distance at z={z} is {d_ang:.2f}")
    return d_ang

def comoving_mpc_to_cm(dl_com, h=0.6774):
    # 1 Mpc = 3.085677581491367e24 cm
    Mpc_to_cm = 3.085677581491367e24
    dl_cm = dl_com / h * Mpc_to_cm
    return dl_cm

def find_snapshot_near(target_z, snap_numbers=None):

    if snap_numbers is None:
        snap_numbers = range(0, 22)

    found_snaps = []
    found_zs = []

    snapshot_dir = "/cosma8/data/dp203/bl267/Data/ClusterSims/L302_N1136_GR/"

    for snap in snap_numbers:
        snap_str = str(snap).zfill(3)
        file_path = os.path.join(snapshot_dir, f"snapdir_{snap_str}", f"snap_{snap_str}.0.hdf5")

        if not os.path.exists(file_path):
            continue

        try:
            with h5py.File(file_path, "r") as f:
                z = f["Header"].attrs["Redshift"]
                found_snaps.append(snap)
                found_zs.append(z)
        except Exception as e:
            print(f"Could not read snapshot {snap}: {e}")

    if not found_snaps:
        raise FileNotFoundError("No snapshots found in given directory.")

    found_snaps = np.array(found_snaps)
    found_zs = np.array(found_zs)

    idx = np.argmin(np.abs(found_zs - target_z))
    return found_snaps[idx], found_zs[idx]

def _randomise_cube(P):
    """Random periodic shift + random flips."""
    Nx, Ny, Nz = P.shape
    P = np.roll(P, np.random.randint(0, Nx), axis=0)
    P = np.roll(P, np.random.randint(0, Ny), axis=1)
    P = np.roll(P, np.random.randint(0, Nz), axis=2)
    if np.random.rand() < 0.5:
        P = np.flip(P, axis=0)
    if np.random.rand() < 0.5:
        P = np.flip(P, axis=1)
    return P

def mosaic_xy(P_e, nx, ny):
    """Tile P_e in x,y to at least nx,ny boxes (random rolls/flips per tile)."""
    Nx, Ny, Nz = P_e.shape
    big = np.zeros((nx*Nx, ny*Ny, Nz), dtype=P_e.dtype)
    for ix in range(nx):
        for iy in range(ny):
            tile = _randomise_cube(P_e.copy())
            x0, y0 = ix*Nx, iy*Ny
            big[x0:x0+Nx, y0:y0+Ny, :] = tile
    return big

class LightCone:
    def __init__(self, simulation, model, realisation, z_edges, delta=500, file_ending="all", fov_deg=2.0, pix_arcmin=0.5):
        self.simulation = simulation
        self.model = model
        self.fileroot = "/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/%s/%s/" % (self.simulation,self.model)
        self.realisation = realisation
        self.delta = delta
        self.file_ending = file_ending

         # --- your instrument / map setup ---
        self.fov_deg   = fov_deg
        self.pix_arcmin = pix_arcmin
        self.fov_rad   = np.deg2rad(fov_deg)
        self.pix_rad   = np.deg2rad(pix_arcmin/60.0)
        self.npix      = int(np.round(self.fov_rad / self.pix_rad))
        print(f"npix = {self.npix}")

        # --- lightcone shells ---
        self.z_edges = z_edges
        self.z_mids  = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        print(f"Shell midpoints: z = {self.z_mids}")


    def load_pressure_grid(self, snap):
        if self.model == "GR" or self.simulation == "L302_N1136":
            pressure_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s_pressure.pickle" % (self.simulation, self.model, self.realisation, snap, self.file_ending)
            print(pressure_dumpfile)
        else:
            pressure_dumpfile = self.fileroot+"pickle_files/%s_%s_%s_s%d_%s_rescaling%s_pressure.pickle" % (self.simulation, self.model, self.realisation, snap, self.file_ending, self.rescaling)

        if os.path.exists(pressure_dumpfile):
            print("%s exists!" % (pressure_dumpfile))
            df = open(pressure_dumpfile, 'rb')
            (P_e, box_size_com, grid_N) = pickle.load(df)
            return P_e, box_size_com, grid_N
        else:
            print("%s does not exist!" % (group_dumpfile))
            sys.exit(0)

    def grid_pressure(self, pressure, positions, volumes, Ngrid, Lbox=301.75):
        # Grid electron pressure
        x = positions[:,0]
        y = positions[:,1]
        z = positions[:,2]

        x = np.mod(x,Lbox)  # Wrap positions into [0, Lbox)
        y = np.mod(y,Lbox)
        z = np.mod(z,Lbox)

        edges = [np.linspace(0, Lbox, Ngrid+1)] * 3  # Bin edges for the grid

        Pe_grid_sum, _ = np.histogramdd(  # Sum pressure into voxels
            sample=np.vstack([x, y, z]).T,
            bins=edges,
            weights=pressure
        )
        print(f"Pressure grid: {Pe_grid_sum}, shape: {Pe_grid_sum.shape}")

    def resample_to_shell_grid(self, P_e0, z_mid, box_size_com, dchi):

        P_e = P_e0 * 1.602e-9 / (3.086e21)**3  # convert from keV kpc^-3 to erg cm^-3

        # geometry
        D_A = angular_diameter_distance(z_mid)             # Mpc
        D_M = (1.0 + z_mid) * D_A                          # comoving Mpc
        Lmap_com = D_M * self.fov_rad                      # comoving Mpc
        nx = max(1, int(np.ceil(Lmap_com / box_size_com)))
        ny = max(1, int(np.ceil(Lmap_com / box_size_com)))

        # mosaic in x,y
        P_big = mosaic_xy(P_e, nx, ny)                     # (nx*Nx, ny*Ny, Nz)
        Nx_big, Ny_big, Nz = P_big.shape
        dz_com = box_size_com / Nz                         # comoving Mpc/h per slice

        # --- how many full boxes and how many slices from the partial one? ---
        n_full = int(dchi // box_size_com)                  # e.g. 2 for 2.5 boxes
        frac  = (dchi / box_size_com) - n_full              # e.g. 0.5
        n_frac_slices = int(np.round(frac * Nz))            # e.g. ~0.5 * Nz
        n_frac_slices = max(0, min(n_frac_slices, Nz))      # clamp

        # --- accumulate 2D projection over the shell thickness ---
        P_accum_2d = np.zeros((Nx_big, Ny_big), dtype=np.float64)

        # full boxes
        for _ in range(n_full):
            Pc = _randomise_cube(P_e)
            # use all Nz slices
            P_accum_2d += np.sum(Pc, axis=2)

        # partial box (take n_frac_slices)
        if n_frac_slices > 0:
            Pc = _randomise_cube(P_e)
            # pick a random starting slice and wrap if needed
            z0 = np.random.randint(0, Nz)
            if z0 + n_frac_slices <= Nz:
                slab = Pc[:, :, z0:z0 + n_frac_slices]
            else:
                # wrap around (periodic)
                end = (z0 + n_frac_slices) - Nz
                slab = np.concatenate([Pc[:, :, z0:], Pc[:, :, :end]], axis=2)
            P_accum_2d += np.sum(slab, axis=2)

        # --- resample to angular pixel grid (npix × npix) ---
        zoom_xy = (self.npix / Nx_big, self.npix / Ny_big)
        P_resampled = zoom(P_accum_2d, zoom=zoom_xy, order=1)

        return P_resampled.astype(np.float64), dchi


    def plot_y_map(self, y_map, output=None):
        npix = y_map.shape[0]
        fov_arcmin = self.fov_deg * 60.0
        extent = [-fov_arcmin/2, fov_arcmin/2, -fov_arcmin/2, fov_arcmin/2]  # arcmin

        plt.figure(figsize=(6,5))
        im = plt.imshow(
            np.log10(y_map + 1e-10),  # log stretch, avoid log(0)
            extent=extent,
            origin="lower",
            cmap="inferno"
        )
        cbar = plt.colorbar(im)
        cbar.set_label(r"$\log_{10}(y)$")

        plt.xlabel("Arcmin")
        plt.ylabel("Arcmin")
        plt.title("Mock SZ y-map")

        if output:
            plt.savefig(output, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {output}")
        else:
            plt.show()


    def calc_y(self):
        """
        Compute the SZ y-map using gas-cell contributions with spherical kernel.
    
        Parameters
        ----------
        positions : array (N,3)
            Gas cell positions in comoving Mpc/h (x,y,z)
        radii : array (N,)
            Gas cell radii in comoving Mpc/h
        pressure : array (N,)
            Gas cell electron pressure in erg/cm^3 (proper)
        z_mid : float
            Redshift of the shell
        """
        # --- initialize y map ---
        y_map = np.zeros((self.npix, self.npix), dtype=np.float64)

        for z_lo, z_hi, z_mid in zip(self.z_edges[:-1], self.z_edges[1:], self.z_mids):

            snap, snap_z = find_snapshot_near(z_mid)

            # --- comoving transverse pixel positions ---
            theta = np.linspace(-self.fov_rad/2, self.fov_rad/2, self.npix)
            phi   = np.linspace(-self.fov_rad/2, self.fov_rad/2, self.npix)
            theta_grid, phi_grid = np.meshgrid(theta, phi)

            # transverse comoving distance
            D_M = (1.0 + z_mid) * angular_diameter_distance(z_mid) * 1000  # kpc
            x_pix = D_M * theta_grid
            y_pix = D_M * phi_grid
            #print(f"x_pix = {x_pix}")

            # scale factor
            a = 1.0 / (1.0 + z_mid)

            pressures, positions, volumes = self.load_pressure_grid(snap)

            # loop over gas cells
            for i in range(len(positions)):
                x0, y0, z0 = positions[i]
                #x0 = positions[i,0]         # comoving kpc
                #y0 = positions[i,1]
                #z0 = positions[i,2]
                R_cell = 2.5 * (3 * volumes[i] / (4 * np.pi))**(1/3)  # kpc
                P_cell = pressures[i]               # not in proper units yet
                R_pix = self.pix_rad * D_M  # kpc
                if R_cell < R_pix:
                    s = R_pix
                else:
                    s = R_cell

                # proper radius and path length conversion
                s_proper = s * a              # proper kpc
                dl_cm_factor = 3.085677581491367e21  # kpc -> cm

                # mask pixels within the projected radius
                r2 = (x_pix - x0)**2 + (y_pix - y0)**2
                mask = r2 <= s_proper**2
                #print(f"r2[mask] = {r2[mask]}")
                if i % 1000 == 0:
                    print(f"i = {i}")
                    print(f"x0 = {x0}")
                    #print(f"r2 = {r2}")
                    print(f"s_proper = {s_proper}")
                if not np.any(mask):
                    continue

                if i % 1000 == 0:
                    print(f"R_cell = {R_cell}")
                    print(f"s = {s}")
                    print(f"r2[mask] = {r2[mask]}")

                # line-of-sight path length through spherical cell
                dl = 2.0 * np.sqrt(s_proper**2 - r2[mask]) * a * dl_cm_factor  # cm

                # Convert units keV kpc^-3 to erg cm^-3
                P_cell = P_cell * 1.6022e-9 / (3.086e21**3)

                # add SZ contribution
                y_map[mask] += (sigma_T / m_e_c2) * P_cell * dl


        # --- plot the map ---
        self.plot_y_map(y_map)

