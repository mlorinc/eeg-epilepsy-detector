import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
from classifier.evolution import feature_chromosome_mapping

def perform_analysis(df: pd.DataFrame, asset_folder: str):
    folder = pathlib.Path(asset_folder)
    folder.mkdir(parents=True, exist_ok=True)

    chrom_counts_df = df.groupby(by="chrom").count().sort_values(by="gen", ascending=False)
    top_10 = chrom_counts_df.head(n=10).index

    # "fit_lat", "fit_spec"
    box_df = pd.melt(df, id_vars=["chrom"], value_vars=["fit_sens"], var_name="metric", value_name="fitness")
    box_df = box_df.loc[box_df["chrom"].isin(top_10)]
    sns.boxplot(data=box_df, x="chrom", y="fitness", hue="metric")
    plt.savefig(folder / "box.pdf")
    plt.savefig(folder / "box.png")
    plt.clf()
    