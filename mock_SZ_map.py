import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import h5py

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


# --- your instrument / map setup ---
fov_deg   = 2.0
pix_arcmin= 0.5
fov_rad   = np.deg2rad(fov_deg)
pix_rad   = np.deg2rad(pix_arcmin/60.0)
npix      = int(np.round(fov_rad / pix_rad))

# --- lightcone shells ---
z_edges = np.linspace(0.0, 2.0, 15)                     # example shells
z_mids  = 0.5 * (z_edges[:-1] + z_edges[1:])

# --- constants in cgs ---
sigma_T = 6.6524587158e-25  # cm^2
m_e_c2  = 8.18710565e-7     # erg
# (If your P_e is in erg/cm^3 and dl in cm, prefactor is sigma_T / m_e_c2)

# --- initialize y map ---
y_map = np.zeros((npix, npix), dtype=np.float32)

for z_lo, z_hi, z_mid in zip(z_edges[:-1], z_edges[1:], z_mids):
    # 1) pick snapshot closest to z_mid
    snap = find_snapshot_near(z_mid)  # <- your function
    # 2) load electron pressure grid for that snapshot (box units & grid)
#    P_e, box_size_com, grid_N = load_pressure_grid(snap)   # (Nx,Ny,Nz), Mpc/h and cell count
    # 3) compute shell comoving thickness
    chi_lo = comoving_distance(z_lo)   # Mpc/h (match units!)
    chi_hi = comoving_distance(z_hi)
#    dchi   = chi_hi - chi_lo
    # 4) resample/tile to angular grid at z_mid
#    P_shell, dl_com = resample_to_shell_grid(P_e, z_mid, fov_rad, pix_rad, box_size_com, dchi)
    # P_shell has shape (npix, npix, Nz_shell), dl_com is comoving thickness per slab
    # 5) convert to y and integrate along LoS
    # Make sure units agree: if P_e is proper, use proper dl;
    # if P_e is comoving, convert appropriately (a factors).
    prefac = sigma_T / m_e_c2       # (cm^2 / erg)
    # Convert comoving Mpc/h to cm, include h and (1+z) if needed (unit bookkeeping!)
#    dl_cm  = comoving_mpc_to_cm(dl_com)  # implement with correct h
#    y_shell = prefac * np.sum(P_shell, axis=2) * dl_cm
#    y_map  += y_shell.astype(np.float32)

# 6) add CMB + noise, 7) convolve with beam, 8) matched filter
# (Use the same matched filter code you already have.)
