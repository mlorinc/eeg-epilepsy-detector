import pandas as pd
import pathlib
import re
from .tf_features import DATA_ROOT

def get_summary_path(patient: str) -> pathlib.Path:
    return DATA_ROOT / patient / f"{patient}-summary.txt"

def parse_seizures(patient: str) -> pd.DataFrame:
    data = []
    with open(get_summary_path(patient)) as file:
        current_file = None
        seizure_count = None
        appended = 0
        start = None
        end = None
        for line in file:
            line = line.strip().lower().replace(":", "")

            if seizure_count != None:
                if line == "":
                    if appended != seizure_count:
                        raise ValueError(f"invalid number of additions got: {appended}, expected: {seizure_count}")
                    seizure_count = None
                    current_file = None
                elif start is None:
                    segments = re.split(r"\s+", line)
                    segments.index("start")
                    time_index = segments.index("time")
                    start = segments[time_index + 1]
                elif end is None:
                    segments = re.split(r"\s+", line)
                    segments.index("end")
                    time_index = segments.index("time")
                    end = segments[time_index + 1]
                    data.append((patient, current_file, int(start), int(end)))
                    appended += 1
                    start = None
                    end = None
                else:
                    raise ValueError("unexpected state")

            if "file name " in line:
                current_file = line[len("file name "):].strip()
            elif "number of seizures in file " in line:
                seizure_count = int(line[len("number of seizures in file "):].strip())
                appended = 0
    df = pd.DataFrame(data=data, columns=["patient", "file", "start", "end"])
    df.sort_values(by=["file", "start"], inplace=True)
    return df

def add_epochs(df_seizure: pd.DataFrame) -> pd.DataFrame:
    df_seizure["first_epoch"] = df_seizure["start"] // 2
    df_seizure["last_epoch"] = df_seizure["end"].apply(lambda e: max(e // 2 - 1, 0) if e % 2 == 0 else e // 2)
    return df_seizure


def get_seizure_summary() -> pd.DataFrame:
    df = pd.DataFrame()
    for i in range(1, 25, 1):
        patient = "chb{:02d}".format(i)
        df_sum = parse_seizures(patient)
        df = pd.concat((df, df_sum))
    df = add_epochs(df)

    debug = df.groupby(by=["patient"]).count().sort_values(by="file", ascending=False)
    print(debug)
    return df