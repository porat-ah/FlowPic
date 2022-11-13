import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import time


def txt_to_lines(file_name, label):
    columns_names = ["Timestamp", "Relative_time", "size", "direction"]
    groups = pd.read_csv(file_name, sep='\t', names=columns_names).groupby('direction')
    attachment = pd.DataFrame(
        {0: [label], 1: [file_name], 2: [np.NaN], 3: [np.NaN], 4: [np.NaN], 5: [np.NaN], 6: [np.NaN]})
    lines = []
    for _dir in range(1):
        df = groups.get_group(_dir)
        line_df = pd.concat(
            [pd.Series([df.shape[0]]), df["Relative_time"] - df["Relative_time"].iloc[0], pd.Series([np.NaN]),
             df["size"]], ignore_index=True)
        line_df = pd.DataFrame({"data": line_df, "columns": line_df.index + 7, "index": np.zeros_like(line_df)})
        line_df = line_df.pivot(index="index", columns="columns", values="data")
        line_df = pd.concat([attachment, line_df], axis=1)
        lines.append(line_df)
    return lines


def main():
    print("creating csv file ...")
    os.chdir("..\QUIC Dataset\pretraining")
    start = time.time()
    for _dir in Path(".").glob('*'):
        print("working on files in", _dir)
        lines = []
        label = re.sub(" ", "", str(_dir))
        for file_name in Path(_dir).glob('*.txt'):
            lines += txt_to_lines(file_name, label)
        if len(lines) > 0:
            df = pd.concat(lines, ignore_index=True)
            df.to_csv(f"{_dir}\{_dir}.csv", index=False, header=False)
    print("running time: ", time.time() - start)


if __name__ == "__main__":
    main()
