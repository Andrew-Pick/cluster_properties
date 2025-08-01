from cluster_properties import SimulationTesting

sim_test = SimulationTesting(
    simulation="L302_N1136",
    models=["GR","F60","F55","F50","F45","F40"],
    realisations=["1","1","1","1","1","1"],
    snapshot=12,
    file_ending="all",
    labels=["L302-N1136-GR", "L302-N1136-F60", "L302-N1136-F55",
            "L302-N1136-F50", "L302-N1136-F45", "L302-N1136-F40"],
    colors=["black","blue","magenta","green","orange","red"],
    defaults=[],
    plot_name="cluster_comparison_allModels"
    show_spread=True
)
sim_test.tng_observables()   # generates summary comparison plots
