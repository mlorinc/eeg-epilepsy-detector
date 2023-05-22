import pandas as pd
import seaborn as sns
import pathlib
import matplotlib.pyplot as plt
from classifier.evolution import feature_chromosome_mapping, psd_features_end

def get_category(chromosome: str):
    if "1" in chromosome[:psd_features_end] and "1" in chromosome[psd_features_end:]:
        return "mixed"
    elif "1" in chromosome[:psd_features_end]:
        return "psd"
    else:
        return "epoch"

def get_feature_distribution(df: pd.DataFrame) -> pd.DataFrame:
    feature_distribution_df = pd.melt(df, id_vars=["chrom"], value_vars=feature_chromosome_mapping, var_name="Feature", value_name="flag")
    feature_distribution_df = feature_distribution_df.loc[feature_distribution_df["flag"] == True]
    feature_counts_df = feature_distribution_df.groupby(by="Feature").count()
    feature_counts_df.rename(columns={"chrom": "count"}, inplace=True)
    feature_counts_df.sort_values(by="count", ascending=False, inplace=True)
    return feature_counts_df, feature_distribution_df

def perform_analysis(df: pd.DataFrame, asset_folder: str):
    df.rename(columns={"fit_sens": "Sensitivity", "fit_lat": "Latency [s]", "fit_spec": "Specificity"}, inplace=True)
    plt.figure(figsize=(16,9))
    folder = pathlib.Path(asset_folder)
    folder.mkdir(parents=True, exist_ok=True)

    df = df.groupby(by="chrom").median()
    df.reset_index(inplace=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    df["Feature type"] = df["chrom"].apply(get_category)
    for i, fitness in enumerate(["Sensitivity", "Latency [s]", "Specificity"]):
        sns.boxplot(data=df, x="Feature type", y=fitness, ax=axes[i], palette="deep")
    fig.savefig(folder / f"chrom_fitness_boxplot.pdf")
    fig.savefig(folder / f"chrom_fitness_boxplot.png")
    fig.clf()
    
    for i, feature in enumerate(feature_chromosome_mapping):
        df[feature] = df["chrom"].apply(lambda chrom: chrom[i] == "1")
    
    feature_counts_df, _ = get_feature_distribution(df)
    sns.barplot(data=feature_counts_df, y=feature_counts_df.index.values, x="count", palette="deep")
    plt.savefig(folder / f"feature_distribution.pdf")
    plt.savefig(folder / f"feature_distribution.png")
    plt.clf()

    success_df = df.sort_values(by="Sensitivity", ascending=False)
    top75 = success_df.loc[success_df["Sensitivity"] >= 0.75]
    feature_counts_df, _ = get_feature_distribution(top75)
    sns.barplot(data=feature_counts_df, y=feature_counts_df.index.values, x="count", palette="deep")
    plt.savefig(folder / f"top_025_feature_distribution.pdf")
    plt.savefig(folder / f"top_025_feature_distribution.png")
    plt.clf()

    feature_counts_df, _ = get_feature_distribution(success_df.head(n=50))
    sns.barplot(data=feature_counts_df, y=feature_counts_df.index.values, x="count", palette="deep")
    plt.savefig(folder / f"top_50_feature_distribution.pdf")
    plt.savefig(folder / f"top_50_feature_distribution.png")
    plt.clf()

    feature_distribution_df = pd.melt(df, id_vars=["chrom", "Sensitivity", "Latency [s]", "Specificity"], value_vars=feature_chromosome_mapping, var_name="Feature", value_name="flag")
    feature_distribution_df = feature_distribution_df.loc[feature_distribution_df["flag"] == True]
    for i, fitness in enumerate(["Sensitivity", "Latency [s]", "Specificity"]):
        sns.boxplot(data=feature_distribution_df, x="Feature", y=fitness, palette="deep")
        plt.xticks(rotation=90)
        plt.subplots_adjust(bottom=0.3)
        plt.savefig(folder / f"metric_{fitness}_boxplot.pdf")
        plt.savefig(folder / f"metric_{fitness}_boxplot.png")
        plt.clf()
