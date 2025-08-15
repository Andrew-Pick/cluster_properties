import numpy as np
from astropy.cosmology import FlatLambdaCDM
from mock_SZ_map import find_snapshot_near

def comoving_distance(z):
    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3089)
    d_comoving = cosmo.comoving_distance(z).value  # in Mpc
    return d_comoving


snap, z = find_snapshot_near(10)
print(f"Closest snapshot: {snap} (z = {z})")
print(comoving_distance(3))
