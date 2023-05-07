import pandas as pd
import features as features
import matplotlib.pyplot as plt 
from sklearn import metrics
from typing import List
from classifier.model import load_model
from classifier.train import test_files
import features.summary as summary
import pathlib

figures = pathlib.Path("figures")

def get_seizure_counts(patient: str):
    seizure_summary = summary.get_seizure_summary()
    counter_df = seizure_summary.groupby(by=["patient", "file"]).count()

    patients = seizure_summary["patient"].unique()
    print(patients)
    print(counter_df.loc[(patient)])
    return counter_df.loc[(patient)]

def merge_epochs(df: pd.DataFrame):
    mask = df["first_epoch"].diff().gt(1)
    group_id = mask.cumsum()
    df = df[~group_id.duplicated()]
    return df

def evaluate(model_name: str, patient: str, labels: List[str]):
    seizure_summary = get_seizure_counts(patient)

    df = pd.read_csv(features.FEATURE_ROOT / patient / features.FEATURE_FILENAME)
    df_test = df.loc[df["file"].isin(test_files), labels + ["class"]]
    df_test_all = df.loc[df["file"].isin(test_files)]


    model = load_model(model_name)
    y = model.predict(df_test[labels])
    df_test_all["pred"] = y
    print(len(df_test_all.loc[df_test_all["class"] == df_test_all["pred"]].index) / len(df_test_all.index))
    print(df_test_all.loc[df_test_all["class"] == 1])

    tp = df_test_all.loc[(df_test_all["class"] == 1) & (df_test_all["pred"] == 1), ["first_epoch", "class", "seizure_start"]]
    fp = df_test_all.loc[(df_test_all["class"] == -1) & (df_test_all["pred"] == 1), ["first_epoch", "class", "seizure_start"]]
    tn = df_test_all.loc[(df_test_all["class"] == -1) & (df_test_all["pred"] == -1), ["first_epoch", "class", "seizure_start"]]
    fn = df_test_all.loc[(df_test_all["class"] == 1) & (df_test_all["pred"] == -1), ["first_epoch", "class", "seizure_start"]]

    tp = merge_epochs(tp)
    fn = merge_epochs(fn)
    tp["latency"] = (tp["first_epoch"] - tp["seizure_start"]) * 2 

    y_true = df_test_all["class"]
    y_pred = df_test_all["pred"]

    confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=["normal", "seizure"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["normal", "seizure"]) 
    cm_display.plot()
    figures.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures / "confusion.png")
    plt.savefig(figures / "confusion.pdf")
    plt.clf()

    fig, ax_det = plt.subplots(1, 1, figsize=(11, 5))
    metrics.DetCurveDisplay.from_predictions(y_true, y_pred, pos_label=1, ax=ax_det, name="Seizure detection")
    ax_det.set_title("Detection Error Tradeoff (DET)")
    ax_det.grid(linestyle="--")
    fig.savefig(figures / "det.png")
    fig.savefig(figures / "det.pdf")

    print(tp)