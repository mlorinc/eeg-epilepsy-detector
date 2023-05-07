import pandas as pd
import features as features
from typing import List
from classifier.model import create_model, save_model

test_files = ["chb12_27.edf", "chb12_29.edf", "chb12_38.edf", "chb12_42.edf"]

def train(model_filename: str, patient: str, labels: List[str]):
    df = pd.read_csv(features.FEATURE_ROOT / patient / features.FEATURE_FILENAME)
    df_train = df.loc[~df["file"].isin(test_files), labels + ["class"]]
    df_test = df.loc[df["file"].isin(test_files), labels + ["class"]]
    df_test_all = df.loc[df["file"].isin(test_files)]

    model = create_model()
    model = model.fit(df_train[labels].values, df_train["class"])
    save_model(model, model_filename)