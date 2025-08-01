from cluster_properties import SimulationTesting

sim_test = SimulationTesting(
    simulation="L302_N1136",
    models=["GR"],
    realisations=["1"],
    snapshot=12,
    file_ending="highMass_paper",
    labels=["GR"],
    colors=["black"],
    defaults=["Default config"],
    plot_name="cluster_comparison"
)
sim_test.tng_observables()   # generates summary comparison plots
