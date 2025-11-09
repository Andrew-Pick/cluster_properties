import numpy as np
from astropy.cosmology import FlatLambdaCDM
from test import LightCone
from test import comoving_distance
from mock_SZ_map import find_snapshot_near
import pickle
import matplotlib.pyplot as plt

z_edges = np.array([0, 0.2, 0.5, 1, 1.5, 2, 3])
#z_edges = np.array([2, 3])
model = "GR"

lc = LightCone(
    simulation="L302_N1136",
    model=model,
    realisation="1",
    z_edges=z_edges,
    file_ending="all",
    fov_deg=2.0,
    pix_arcmin=0.2344 #0.1172
)

#snap, z = find_snapshot_near(1.5)
#print(f"Closest snapshot: {snap} (z = {z})")
#print(comoving_distance(3))

#y_map = lc.calc_y() #save_y_map="/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/L302_N1136/GR/pickle_files/y_map_no_overlaps_s0.pkl")
if model=="GR":
    fileroot = f"/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/L302_N1136/GR/pickle_files/y_map_no_overlaps_full_fov2.pkl"
else:
    fileroot = f"/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/L302_N1136/GR/pickle_files/y_map_no_overlaps_full_fov2_{model}.pkl"
with open(fileroot, "rb") as f:
    y_map = pickle.load(f)
print(f'Loaded file {fileroot}')

#fileroot = f"/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/L302_N1136/GR/pickle_files/y_map_no_overlaps_full_fov2_{model}.pkl"
#with open(fileroot, "wb") as f:
#    pickle.dump((y_map), f)
#print(f'Data saved at {fileroot}')

snr = lc.signal_noise_ratio(y_map)
y_coords, y_peaks = lc.find_peaks(y_map, 10**(-5.5))
snr_coords, snr_peaks = lc.find_peaks(snr, 0)
lc.plot_y_map(y_map, output=None) #f'no_overlaps_full_fov2_{model}')
#lc.plot_y_map(snr, output=None, log=False)

pos, M = lc.cluster_positions()
#print(f'cluster positions = {pos}')
#print(f'cluster masses = {M}')
#print(f'Number of clusters in fov = {M.shape}')

y_coords = y_coords[:, [1, 0]].astype(int)
#print(f'peak coordinates(x_pix, y_pix) = {y_coords}')
#print(f'peak values = {y_peaks}')
snr_coords = snr_coords[:, [1, 0]].astype(int)
#print(f'Number of peaks = {snr_coords.shape[0]}')
#print(f'snr peak coordinates(x_pix, y_pix) = {snr_coords}')
#print(f'snr peak values = {snr_peaks}')
pos_xy = pos[:,0:2].astype(int)
#print(f'pos_xy = {pos_xy}')

matches = np.isclose(snr_coords[:, None, :], pos_xy[None, :, :], atol=1).all(-1)
#matches = (snr_coords[:, None, :] == pos_xy[None, :, :]).all(-1)

# Get indices of matches
peak_idx, mass_idx = np.where(matches)
pairs = np.column_stack((peak_idx, mass_idx))
#print(pairs)
#print(f'Number of matches = {pairs.shape[0]}')

y = snr_peaks[peak_idx]
x = M[mass_idx]

plt.scatter(x, y)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$M_{500}$")
plt.ylabel(r"$\xi$")
plt.show()
