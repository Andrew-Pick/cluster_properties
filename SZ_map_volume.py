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

def randomise_particles(self, pos, Lbox=301.75):
    """Apply random periodic shift and random flips to particle positions."""
    pos_new = pos.copy()

    # random shifts
    shifts = np.random.uniform(0, Lbox, size=3)
    pos_new = (pos_new + shifts) % Lbox

    # random flips
    for axis in range(3):
        if np.random.rand() < 0.5:
            pos_new[:,axis] = Lbox - pos_new[:,axis]

    return pos_new

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


    def splat_to_grid(self, positions, pressures, volumes, z, Lbox=301.75): 
        """
        Project scattered gas cells onto the 2D y-map grid using a top-hat kernel.
        positions : (N,3) positions in comoving Mpc/h
        pressures : (N,) electron pressure [erg/cm^3]
        volumes   : (N,) cell volumes [Mpc^3/h^3]
        z         : redshift of the shell
        """
        # --- initialize empty map ---
        y_map = np.zeros((self.npix, self.npix), dtype=np.float64)

        # pixel angular size in radians
        dtheta = self.pix_rad
        pix_coords = (np.arange(self.npix) - self.npix/2 + 0.5) * dtheta
        theta_x, theta_y = np.meshgrid(pix_coords, pix_coords, indexing="ij")

        # loop over cells
        for (x, y, zpos), P_e, V in zip(positions, pressures, volumes):
            if zpos <= 0:
                continue

            # projected angular coords
            ang_x = x / zpos
            ang_y = y / zpos

            # central pixel
            i_pix = int(ang_x / dtheta + self.npix/2)
            j_pix = int(ang_y / dtheta + self.npix/2)
            if i_pix < 0 or i_pix >= self.npix or j_pix < 0 or j_pix >= self.npix:
                continue

            # effective cell radius
            r_cell = 2.5 * ((3*V)/(4*np.pi))**(1/3)
            r_pix  = zpos * dtheta
            s      = min(r_cell, r_pix)

            # pixel radius in index units
            pix_radius = int(np.ceil(s / (zpos*dtheta)))

            # distribute contribution to nearby pixels
            for di in range(-pix_radius, pix_radius+1):
                for dj in range(-pix_radius, pix_radius+1):
                    ii, jj = i_pix+di, j_pix+dj
                    if ii < 0 or ii >= self.npix or jj < 0 or jj >= self.npix:
                        continue
                    # angular distance from pixel center
                    dx = theta_x[ii,jj] - ang_x
                    dy = theta_y[ii,jj] - ang_y
                    dist = zpos * np.sqrt(dx**2 + dy**2)
                    if dist < s:
                        # top-hat kernel weight
                        weight = 1.0
                        y_map[ii,jj] += P_e * V * weight / (zpos**2)

        # prefactor to convert to y
        prefac = sigma_T / m_e_c2  # (cm^2 / erg)
        return prefac * y_map


def resample_shell_particles(self, positions, pressures, volumes, z_mid, dchi, Lbox):
    """
    Make a projected SZ map for a redshift shell using kernel splatting on particles.

    Parameters
    ----------
    positions : (N,3) ndarray
        Particle positions [Mpc/h, comoving]
    pressures : (N,) ndarray
        Electron pressures [erg/cm^3]
    volumes   : (N,) ndarray
        Cell volumes [Mpc^3/h^3]
    z_mid : float
        Midpoint redshift of shell
    dchi : float
        Comoving thickness of the shell [Mpc/h]
    Lbox : float
        Size of simulation box [Mpc/h]
    """

    # --- geometry ---
    D_A = angular_diameter_distance(z_mid)        # [Mpc]
    D_M = (1.0 + z_mid) * D_A                     # comoving [Mpc]
    Lmap_com = D_M * self.fov_rad                 # transverse comoving size
    nx = max(1, int(np.ceil(Lmap_com / Lbox)))    # number of tiles in x
    ny = max(1, int(np.ceil(Lmap_com / Lbox)))    # number of tiles in y

    # --- how many boxes in z (radial) ---
    n_full = int(dchi // Lbox)        # full boxes
    frac   = (dchi / Lbox) - n_full   # leftover fraction
    n_extra = n_full + (1 if frac > 0 else 0)

    # --- initialize shell map ---
    y_shell = np.zeros((self.npix, self.npix), dtype=np.float64)

    # --- loop over boxes along z ---
    for iz in range(n_extra):
        # randomize particle positions for each box copy
        pos_rand = self.randomise_particles(positions, Lbox)

        # if this is a fractional box, randomly select a subset along z
        if (iz == n_full) and (frac > 0):
            z_extent = frac * Lbox
            mask = (pos_rand[:,2] < z_extent)
            pos_rand = pos_rand[mask]
            pressures_sub = pressures[mask]
            volumes_sub   = volumes[mask]
        else:
            pressures_sub = pressures
            volumes_sub   = volumes

        # --- loop over x–y tiles ---
        for ix in range(nx):
            for iy in range(ny):
                # shift positions to tile
                pos_tile = pos_rand.copy()
                pos_tile[:,0] += ix * Lbox
                pos_tile[:,1] += iy * Lbox

                # splat into map
                y_shell += self.splat_to_grid(pos_tile, pressures_sub, volumes_sub, z_mid)

    return y_shell


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

        # --- initialize y map ---
        y_map = np.zeros((self.npix, self.npix), dtype=np.float32)

        for z_lo, z_hi, z_mid in zip(self.z_edges[:-1], self.z_edges[1:], self.z_mids):
            # 1) pick snapshot closest to z_mid
            snap, snap_z = find_snapshot_near(z_mid)
            # 2) load electron pressure grid for that snapshot (box units & grid)
            P_e, pos, vol = self.load_pressure_grid(snap)   # (Nx,Ny,Nz), Mpc/h and cell count
            # 3) compute shell comoving thickness
            chi_lo = comoving_distance(z_lo)   # Mpc/h (match units!)
            chi_hi = comoving_distance(z_hi)
            dchi   = chi_hi - chi_lo
            # 4) resample/tile to angular grid at z_mid
#            P_shell, dchi_com = self.resample_to_shell_grid(P_e, z_mid, box_size_com, dchi)
            # P_shell has shape (npix, npix, Nz_shell), dl_com is comoving thickness per slab
            # 5) convert to y and integrate along LoS
            
            # Convert comoving Mpc/h to cm, include h and (1+z) if needed (unit bookkeeping!)
            # If P_e is comoving pressure: P_proper = P_com * (1+z)^3
            # Proper path length: dl_proper = dchi_com / (1+z)
#            a = 1.0 / (1.0 + z_mid)

#            P_proper = P_shell * (1.0/a**3)  # If pressure is comoving
#            P_proper = P_shell  # If pressure is proper

#            dl_cm = comoving_mpc_to_cm(dchi_com) * a     # cm (proper)
            dl_cm  = comoving_mpc_to_cm(dchi_com)
            # Make sure units agree: if P_e is proper, use proper dl;
            # if P_e is comoving, convert appropriately (a factors).

            # project shell using kernel splatting + mosaics + z-stacking
            y_shell = self.resample_shell_particles(pos, P_e, vol, z_mid, dchi, Lbox=301.75)

            # add to total y-map
            y_map += y_shell


        self.plot_y_map(y_map)

        # 6) add CMB + noise, 7) convolve with beam, 8) matched filter
        # (Use the same matched filter code you already have.)
