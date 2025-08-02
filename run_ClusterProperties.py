from cluster_properties import ClusterProperties

cp = ClusterProperties(
    simulation="L302_N1136",
    model="F40",
    realisation="1",
    snapshot=12,
    mass_cut=1e13,
    delta=500,
    file_ending="all",
    rescaling="true",
    core_frac=0.15
)
cp.cluster_properties()    # compute profiles & properties
cp.profile()               # compute median profiles
cp.proxy_scaling_relation()  # scaling relations
