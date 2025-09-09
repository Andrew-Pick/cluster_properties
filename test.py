import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import h5py
import pickle
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

#np.random.seed(1273)

# --- constants in cgs ---
sigma_T = 6.6524587158e-25  # cm^2
m_e_c2  = 8.18710565e-7     # erg
# (If your P_e is in erg/cm^3 and dl in cm, prefactor is sigma_T / m_e_c2)

# --- cosmology helpers (you can use astropy.cosmology instead) ---
def comoving_distance(z):
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    d_comoving = cosmo.comoving_distance(z).value  # in Mpc
    print(f"Comoving distance at z={z} is {d_comoving:.2f}")
    return d_comoving # * 0.6774

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

def _rand_shift_flip_3d(pos, L):
    """
    Apply a single random periodic shift and an optional flip around L/2.
    Vectorized for speed. Returns transformed coord in [0, L).
    """
    pos = np.array(pos, copy=True)

    # random uniform shift in [0,L)
    s = np.random.uniform(0, L, size=3)
    # optional flip about center
    if np.random.rand() < 0.5:
        pos[:,0] = (L - pos[:,0])
    if np.random.rand() < 0.5:
        pos[:,1] = (L - pos[:,1])
    if np.random.rand() < 0.5:
        pos[:,2] = (L - pos[:,2])
    pos = (pos + s) % L
    return pos


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
        self.Lbox = 301.75 * 1000 / 0.6774  # kpc

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
        extent = [0, fov_arcmin, 0, fov_arcmin]  # arcmin

        plt.figure(figsize=(6,5))
        im = plt.imshow(
            np.log10(y_map + 1e-7),  # log stretch, avoid log(0)
            extent=extent,
            origin="lower",
            cmap="viridis",
#            vmin=-7,
#            vmax=-4
        )
        cbar = plt.colorbar(im)
        cbar.set_label(r"$\log_{10}(y)$")

        plt.xlabel("Arcmin")
        plt.ylabel("Arcmin")
        plt.title("Mock SZ y-map")

        if output:
            directory = os.path.dirname("/cosma/home/dp203/dc-pick1/cluster_properties/plots/SZ_maps/")
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"/cosma/home/dp203/dc-pick1/cluster_properties/plots/SZ_maps/{output}.pdf", dpi=200, bbox_inches="tight")
            print(f"Saved figure to {output}")
        else:
            plt.show()


    def calc_y(self, output=None):
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
        Lbox = self.Lbox

        # --- initialize y map ---
        y_map = np.zeros((self.npix, self.npix), dtype=np.float64)

        for z_lo, z_hi, z_mid in zip(self.z_edges[:-1], self.z_edges[1:], self.z_mids):

            snap, snap_z = find_snapshot_near(z_mid)

            pressures, positions, volumes = self.load_pressure_grid(snap)

            x = positions[:,0] % Lbox
            y = positions[:,1] % Lbox
            z = positions[:,2] % Lbox

            # scale factor
            a = 1.0 / (1.0 + z_mid)

            # transverse comoving distance
            D_A = angular_diameter_distance(z_mid) * 1000  # kpc
            D_M = (1.0 + z_mid) * D_A  # kpc

            # FOV in comoving kpc at shell
            Lmap_com_kpc = D_M * self.fov_rad
            print(f"Lmap = {Lmap_com_kpc}")

            # How many tiles in x,y?
            nx = max(1, int(np.ceil(Lmap_com_kpc / Lbox)))
            ny = max(1, int(np.ceil(Lmap_com_kpc / Lbox)))

            # --- comoving transverse pixel positions ---
            theta = np.linspace(0, self.fov_rad, self.npix)
            phi   = np.linspace(0, self.fov_rad, self.npix)
            theta_grid, phi_grid = np.meshgrid(theta, phi)
            x_pix = D_M * theta_grid
            y_pix = D_M * phi_grid

            # Flatten into a (Npix^2, 2) array of pixel centers
            pix_coords = np.vstack([x_pix.ravel(), y_pix.ravel()]).T
            tree = cKDTree(pix_coords)

            R_pix = self.pix_rad * D_M  # comoving kpc
#            print(f"R_pix = {R_pix}")

            # Shell comoving thickness in *kpc* (comoving)
            chi_lo = comoving_distance(z_lo) * 1000  # kpc
            chi_hi = comoving_distance(z_hi) * 1000  # kpc
            dchi = (chi_hi - chi_lo)  # kpc

            # LOS tiling counts
            n_full = int(dchi // Lbox)
            frac   = (dchi / Lbox) - n_full
            partial_thickness = max(0.0, min(frac * Lbox, Lbox))

            # Pre-compute cell radius from volume (in kpc^3 proper)
            R_cell_kpc = 2.5 * (3.0 * np.maximum(volumes, 0.0) / (4.0 * np.pi))**(1.0/3.0)  # proper kpc


            def add_to_y_map(P, pos, R, vol):
                # loop over gas cells
                for i in range(len(pos)):
                    x0, y0, z0 = pos[i] / a  # comoving kpc
                    R_cell = R[i] / a  # comoving kpc
                    if R_cell < R_pix:
                        s = R_pix  # comoving
                    else:
                        s = R_cell

                    # use s^2 instead of the volume from data - removes need for dl in sum
                    P_cell = P[i] * vol[i] / (s * a)**2  # proper keV kpc^-2

                    if i % 50000 == 0:
                        print("Processing...")

                    # proper radius and path length conversion
#                    s_proper = s * a              # proper kpc
#                    dl_cm_factor = 3.085677581491367e21  # kpc -> cm

                    # find all pixel centers within radius R
                    idxs = tree.query_ball_point([x0, y0], r=s)
                    if not idxs:
                        continue

                    # mask pixels within the projected radius
                    pix_xy = pix_coords[idxs]  # comoving kpc
                    r2 = (pix_xy[:,0] - x0)**2 + (pix_xy[:,1] - y0)**2
#                    r2 = (x_pix - x0)**2 + (y_pix - y0)**2
                    mask = r2 <= s**2
                    #print(f"r2[mask] = {r2[mask]}")
                    if i % 50000 == 0:
                        print(f"z_mid = {z_mid}")
                        print(f"i = {i}")
                        #print(f"x0 = {x0}")
                        #print(f"r2 = {r2}")
                        #print(f"s_proper = {s_proper}")
                    if not np.any(mask):
                        continue

                    if i % 50000 == 0:
                        print(f"R_cell = {R_cell}")
                        print(f"s = {s}")
                        print(f"r2[mask] = {r2[mask]}")

                    # line-of-sight path length through spherical cell
#                    dl = 2.0 * np.sqrt(s**2 - r2[mask]) * dl_cm_factor * a  # proper kpc

                    # Convert units keV kpc^-3 to erg cm^-3
                    P_cell = P_cell * 1.6022e-9 / (3.086e21**2)  # given in proper units

                    # add SZ contribution
                    flat_idxs = np.array(idxs)[mask]
                    y_map.ravel()[flat_idxs] += (sigma_T / m_e_c2) * P_cell #* dl  # proper

#                    print(f"ymap = {y_map.ravel()[flat_idxs]}")

            # full boxes
            for _ in range(n_full):
                pos = _rand_shift_flip_3d(positions, Lmap_com_kpc * a)
                print(f"min pos = {np.min(pos)}")
                add_to_y_map(pressures, pos, R_cell_kpc, volumes)

            # partial box (take n_frac_slices)
            if partial_thickness > 0:
                pos = _rand_shift_flip_3d(positions, Lmap_com_kpc * a)
                zpos = pos[:, 2]

                # pick a random starting slice and wrap if needed
                z0 = np.random.uniform(0, Lbox * a)
                zsel = ((zpos >= z0) & (zpos < z0 + partial_thickness)) \
                       if z0 + partial_thickness <= Lbox else \
                       ((zpos >= z0) | (zpos < (z0 + partial_thickness - Lbox)))
                pos_partial = pos[zsel]
                P_partial   = pressures[zsel]
                R_partial   = R_cell_kpc[zsel]
                V_partial   = volumes[zsel]
                add_to_y_map(P_partial, pos_partial, R_partial, V_partial)

        # --- plot the map ---
        self.plot_y_map(y_map, output=output)

