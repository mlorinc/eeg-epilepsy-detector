import numpy as np
import pandas as pd
import features.tf_features as features
import features.summary as summary
import numpy as np
from classifier.evolution import NSGA, feature_chromosome_mapping
from classifier.model import create_model, save_model, load_model
from classifier.train import test_files
import stats
import pathlib
import argparse
import glob

SUMMARY_FILE = "summary.csv.gz"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Feature optimization for KNN classifier using NSGA II")

    subparsers = parser.add_subparsers(dest="command", title="subcommands")

    extract_parser = subparsers.add_parser("features:extract", help="extract features from EEG data")
    extract_parser.add_argument("--window-size", type=int, default=3, help="size of the sliding window")
    extract_parser.add_argument("patient", type=int, nargs="+", help="patient number")

    optimize_parser = subparsers.add_parser("features:optimize", help="find optimal features using EA")
    optimize_parser.add_argument("experiment_folder", type=str, help="experiment folder where optimization statistics will be stored")
    optimize_parser.add_argument("patient", type=int, help="patient number")

    analyze_parser = subparsers.add_parser("features:analyze", help="analyze results of EA optimization")
    analyze_parser.add_argument("experiment_folder", type=str, help="experiment folder where statistics data are located")
    analyze_parser.add_argument("assets_folder", type=str, help="assets folder where generated figures and tables will be stored")
    return parser, parser.parse_args()

def extract(args):
    features.create_patients_feature_dataset(args.window, summary.get_seizure_summary(), args.patient)

def optimize(args):
    df = pd.read_csv(features.FEATURE_ROOT / f"chb{args.patient}" / features.FEATURE_FILENAME)
    df_train = df.loc[~df["file"].isin(test_files)]
    df_test = df.loc[df["file"].isin(test_files)]

    # Perform dry test run with extremely small dataset
    # df_pos = df_train.loc[df_train["class"] == 1, :][0:2]
    # df_neg = df_train.loc[df_train["class"] == -1, :][0:2]

    # df_test_pos = df_test.loc[df_test["class"] == 1, :][0:2]
    # df_test_neg = df_test.loc[df_test["class"] == -1, :][0:2]

    # df_train = pd.concat((df_pos, df_neg))
    # df_test = pd.concat((df_test_pos, df_test_neg))

    # P * G <= 420
    out_folder = pathlib.Path(args.experiment_folder or "experiments")
    out_folder.mkdir(parents=True, exist_ok=False)
    for i in range(1):
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

def get_nsga_number(file: str) -> int:
    start = file.index(".") + 1
    end = file.index(".", start)
    return int(file[start:end])

def analyze(args):
    folder = pathlib.Path(args.experiment_folder)
    summary_file = folder / SUMMARY_FILE

    if summary_file.exists():
        df = pd.read_csv(summary_file)
    else:
        globs = glob.glob("*.csv.gz", root_dir=folder)
        globs.sort(key=get_nsga_number)

        for i, file in enumerate(globs):
            print(f"merging {file} into {summary_file}")
            df = pd.read_csv(folder / file)
            df["run"] = i + 1
            if df.empty:
                raise ValueError(f"unexpected empty report in {file}")
            if i == 0:
                df.to_csv(summary_file, mode="w", header=True, index=False)
            else:
                df.to_csv(summary_file, mode="a", header=False, index=False)
        # repeat attempt
        return analyze(args)
    stats.perform_analysis(df, args.assets_folder)
    

def main():
    parser, args = parse_arguments()
    if args.command == "features:extract":
        print(f"Extracting features for patients {', '.join(args.patient)} with window size {args.window_size}")
        extract(args)
    elif args.command == "features:optimize":
        print(f"Optimizing features for patient {args.patient} in folder {args.experiment_folder}")
        optimize(args)
    elif args.command == "features:analyze":
        print(f"Analyzing result of EA in folder {args.experiment_folder}")
        analyze(args)
    else:
        parser.print_help()
        exit(1)
    exit(0)

main()