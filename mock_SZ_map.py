import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import h5py
import pickle
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

# --- constants in cgs ---
sigma_T = 6.6524587158e-25  # cm^2
m_e_c2  = 8.18710565e-7     # erg
# (If your P_e is in erg/cm^3 and dl in cm, prefactor is sigma_T / m_e_c2)

# --- cosmology helpers (you can use astropy.cosmology instead) ---
def comoving_distance(z):
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    d_comoving = cosmo.comoving_distance(z).value  # in Mpc
    print(f"Comoving distance at z={z} is {d_comoving:.2f}")
    return d_comoving

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

class LightCone:
    def __init__(self, simulation, model, realisation, delta=500, file_ending="all", fov_deg=2.0, pix_arcmin=0.5, zmin=0, zmax=3, znum=6):
        self.simulation = simulation
        self.model = model
        self.fileroot = "/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/%s/%s/" % (self.simulation,self.model)
        self.realisation = realisation
        self.delta = delta
        self.file_ending = file_ending

         # --- your instrument / map setup ---
        self.fov_deg   = 2.0
        self.pix_arcmin = 0.5
        self.fov_rad   = np.deg2rad(fov_deg)
        self.pix_rad   = np.deg2rad(pix_arcmin/60.0)
        self.npix      = int(np.round(self.fov_rad / self.pix_rad))

        # --- lightcone shells ---
        self.z_edges = np.linspace(zmin, zmax, znum)
        self.z_mids  = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        print(f"Shell midpoints: z = {z_mids}")


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


    def resample_to_shell_grid(self, P_e, z_mid, box_size_com, dchi):
        # --- random shifts ---
        shifts = [np.random.randint(0, s) for s in P_e.shape]
        P_e = np.roll(P_e, shifts[0], axis=0)
        P_e = np.roll(P_e, shifts[1], axis=1)
        P_e = np.roll(P_e, shifts[2], axis=2)

        # --- random flips/reflections ---
        for axis in range(3):
            if np.random.rand() < 0.5:
                P_e = np.flip(P_e, axis=axis)

        # --- target grid resolution ---
        self.npix = int(np.round(self.fov_rad / self.pix_rad))

        # Physical angular size of cube at z_mid
        D_ang = angular_diameter_distance(z_mid)  # Mpc
        theta_box = box_size_com / D_ang          # radians

        # How many pixels cover the box in angular space?
        n_box_pix = int(np.round(theta_box / self.pix_rad))

        # --- rescale cube to this angular resolution ---
        zoom_factors = (n_box_pix / P_e.shape[0],
                        n_box_pix / P_e.shape[1],
                        dchi / (box_size_com / P_e.shape[2]))  # scale z to shell thickness

        P_resampled = zoom(P_e, zoom=zoom_factors, order=1)

        # --- center into npix × npix map (pad/crop if necessary) ---
        P_shell = np.zeros((self.npix, self.npix, P_resampled.shape[2]), dtype=np.float32)
        nx, ny, nz = P_resampled.shape
        x0 = (self.npix - nx) // 2
        y0 = (self.npix - ny) // 2
        P_shell[x0:x0+nx, y0:y0+ny, :] = P_resampled

        dl_com = dchi / P_shell.shape[2]

        return P_shell, dl_com


    def plot_y_map(y_map, output=None):
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
            P_e, box_size_com, grid_N = self.load_pressure_grid(snap)   # (Nx,Ny,Nz), Mpc/h and cell count
            # 3) compute shell comoving thickness
            chi_lo = comoving_distance(z_lo)   # Mpc/h (match units!)
            chi_hi = comoving_distance(z_hi)
            dchi   = chi_hi - chi_lo
            # 4) resample/tile to angular grid at z_mid
            P_shell, dl_com = resample_to_shell_grid(P_e, z_mid, box_size_com, dchi)
            # P_shell has shape (npix, npix, Nz_shell), dl_com is comoving thickness per slab
            # 5) convert to y and integrate along LoS
            # Make sure units agree: if P_e is proper, use proper dl;
            # if P_e is comoving, convert appropriately (a factors).
            prefac = sigma_T / m_e_c2       # (cm^2 / erg)
            # Convert comoving Mpc/h to cm, include h and (1+z) if needed (unit bookkeeping!)
            dl_cm  = comoving_mpc_to_cm(dl_com)  # implement with correct h
            y_shell = prefac * np.sum(P_shell, axis=2) * dl_cm
            y_map  += y_shell.astype(np.float32)

            plot_y_map(y_map)

        # 6) add CMB + noise, 7) convolve with beam, 8) matched filter
        # (Use the same matched filter code you already have.)
