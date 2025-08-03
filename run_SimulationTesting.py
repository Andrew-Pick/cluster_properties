from cluster_properties import SimulationTesting

sim_test = SimulationTesting(
    simulation="L302_N1136",
    models=["GR","F40"],
    realisations=["1","1"],
    snapshot=12,
    file_ending="all",
    labels=["L302-N1136-GR", "L302-N1136-F40"],
    colors=["black","red"],
    defaults=[],
    plot_name="test_MyData_F40_GR",
    show_spread=True
)
sim_test.tng_observables()   # generates summary comparison plots
