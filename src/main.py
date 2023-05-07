import mne
import numpy as np
import pandas as pd
import features.tf_features as features
import features.summary as summary
import numpy as np
from classifier.evolution import NSGA, feature_chromosome_mapping
from classifier.model import create_model, save_model, load_model
from classifier.train import test_files
import pathlib

def create_dataset():
    features.create_patients_feature_dataset(3, summary.get_seizure_summary(), [12])

def main():
    df = pd.read_csv(features.FEATURE_ROOT / "chb12" / features.FEATURE_FILENAME)
    df_train = df.loc[~df["file"].isin(test_files)]
    df_test = df.loc[df["file"].isin(test_files)]

    # df_pos = df_train.loc[df_train["class"] == 1, :][0:2]
    # df_neg = df_train.loc[df_train["class"] == -1, :][0:2]

    # df_test_pos = df_test.loc[df_test["class"] == 1, :][0:2]
    # df_test_neg = df_test.loc[df_test["class"] == -1, :][0:2]

    # df_train = pd.concat((df_pos, df_neg))
    # df_test = pd.concat((df_test_pos, df_test_neg))

    # P * G <= 420
    out_folder = pathlib.Path("experiments")
    out_folder.mkdir(parents=True, exist_ok=False)
    for i in range(30):
        nsga = NSGA(22, 20, len(feature_chromosome_mapping), np.inf, create_model, df_train, df_test)

        counter = 0
        while not nsga.create_generation():
            print(f"[{counter}] Sensitivity: {nsga.best_chromosome.sensitivity}")
            print(f"[{counter}] Latency: {nsga.best_chromosome.latency}")
            print(f"[{counter}] Specifity: {nsga.best_chromosome.specifity}")
            print(f"[{counter}] Chromosome: {''.join(map(str, nsga.best_chromosome.chromosome))}")
            counter += 1 
        
        report_df = nsga.get_report()
        report_df.to_csv(out_folder / f"nsga.{i}.csv.gz", index=False, header=True)
        save_model(nsga.best_chromosome.model, out_folder / f"model.{i}.joblib")
        print(report_df)

main()