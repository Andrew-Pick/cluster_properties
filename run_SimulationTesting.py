from cluster_properties import SimulationTesting

sim_test = SimulationTesting(
    simulation="L302_N1136",
    models=["GR"],
    realisations=["1"],
    snapshot=12,
    file_ending="all",
    labels=["L302-N1136-GR"],
    colors=["black"],
    defaults=[],
    plot_name="test_MyData_GR",
    show_spread=True
)
sim_test.tng_observables()   # generates summary comparison plots
