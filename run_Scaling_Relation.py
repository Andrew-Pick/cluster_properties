from scaling_relations import Scaling_Relation

sr = Scaling_Relation(
    simulation="L302_N1136",
    models=["GR"],
    realisations=["1"],
    snapshot=12,
    file_ending="all",
    labels=["L302-N1136-GR"],
    colors=["black"],
    defaults=[],
    plot_name="scaling_relation_GR",
    show_spread=False
)


sr.redshift()
