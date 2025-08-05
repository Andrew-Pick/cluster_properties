from cluster_properties import ClusterProperties
#import pickle
#
#with open("/cosma8/data/dp203/dc-pick1/Projects/Ongoing/Clusters/My_Data/L302_N1136/GR/pickle_files/L302_N1136_GR_1_s12_all.pickle","rb") as f:
#   (M500, M200, Mg500, Mstar, A19_Mstar, SMF,
#    vol_T500, mass_T500, vol_T500_with_core, mass_T500_with_core,
#    Ysz_with_core, Ysz_no_core, Lx_with_core, Lx_no_core,
#    vol_temp_profile, mass_temp_profile, density_profile,
#    electron_pressure_profile, cum_fgas) = pickle.load(f)

cp = ClusterProperties(
    simulation="L302_N1136",
    model="F45",
    realisation="1",
    snapshot=12,
    mass_cut=1e13,
    delta=500,
    file_ending="all",
    rescaling="true",
    core_frac=0.15
)
cp.cluster_properties()    # compute profiles & properties
#cp.proxy_scaling_relation(
#    proxy_type=["T"],
#    temp_weight="mass",
#    no_core=True,
#    use_analytical=True)
