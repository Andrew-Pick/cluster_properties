from cluster_properties import SimulationTesting

sim_test = SimulationTesting(
    simulation="L302_N1136",
    models=["GR","F40","F45","F50","F55","F60"],
    realisations=["1","1","1","1","1","1"],
    snapshot=12,
    file_ending="all",
    labels=["GR","F40","F45","F50","F55","F60"],
    colors=["black","blue","pink","green","yellow","red"],
    defaults=["Default config"],
    plot_name="cluster_comparison_allModels"
)
sim_test.tng_observables()   # generates summary comparison plots
