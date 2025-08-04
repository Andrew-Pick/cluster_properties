from scaling_relations import Scaling_Relation

sr = Scaling_Relation(
    simulation="L302_N1136",
    models=["GR","F60","F55","F50","F45","F40"],
    realisations=["1","1","1","1","1","1"],
    snapshot=12,
    file_ending="all",
    labels=["GR","F60","F55","F50","F45","F40"],
    colors=["black","blue","magenta","green","orange","red"],
    defaults=[],
    plot_name="scaling_relation_allModels",
    show_spread=False
)


sr.redshift()
