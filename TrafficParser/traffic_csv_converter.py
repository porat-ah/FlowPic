#!/usr/bin/env python
"""
Read traffic_csv
"""

import os
import csv

import matplotlib.pyplot as plt

from sessions_plotter import *
import glob
import re

CLASSES_DIR = "../class/**/"
# CLASSES_DIR = "../classes_csvs/**/**/"

FlowPic = False  # True - create FlowPic , False - create miniFlowPic
if FlowPic:
    TPS = 60  # TimePerSession in secs
    DELTA_T = 60  # Delta T between splitted sessions
    MIN_TPS = 50
    MIN_LENGTH = 10
    IMAGE_SIZE = 1500
else:
    TPS = 15  # TimePerSession in secs
    DELTA_T = 15  # Delta T between splitted sessions
    MIN_TPS = 0
    MIN_LENGTH = 100
    IMAGE_SIZE = 32
DEBUG = False
FIRST_15 = True


def export_class_dataset(dataset, class_dir, name=None):
    print("Start export dataset")
    if name is None:
        name = class_dir + "/" + "_".join(re.findall(r"[\w']+", class_dir)[-2:])
    np.save(name, dataset)
    print(dataset.shape)


def traffic_csv_converter(file_path):
    print("Running on " + file_path)
    dataset = []
    if FIRST_15:
        dataset_first_15 = []
    counter = 0
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            length = int(float(row[7]))
            ts = np.array(row[8:8 + length], dtype=float)
            sizes = np.array(row[9 + length:9 + length + length], dtype=float)
            sizes.astype(int)
            if DEBUG:
                print(max(ts))
                plt.scatter(x=ts, y=sizes)
                plt.show()
            if length > MIN_LENGTH:
                for t in range(int(ts[-1] / DELTA_T - TPS / DELTA_T) + 1):
                    mask = ((ts >= t * DELTA_T) & (ts <= (t * DELTA_T + TPS)))
                    ts_mask = ts[mask]
                    sizes_mask = sizes[mask]
                    if DEBUG:
                        print("mask length =", len(ts_mask), "range =", ts_mask[-1] - ts_mask[0])
                    if len(ts_mask) > MIN_LENGTH and ts_mask[-1] - ts_mask[0] > MIN_TPS:
                        h = session_2d_histogram(ts_mask, sizes_mask, DEBUG)
                        dataset.append([h])
                        if FIRST_15 and t == 0:
                            dataset_first_15.append([h])
                        counter += 1
                        if counter % 100 == 0:
                            print(counter)
    if FIRST_15:
        class_dir = file_path.split('\\')
        class_dir = '\\'.join(class_dir[:-1]) + '\\'
        name = class_dir + "/" + "_".join(re.findall(r"[\w']+", class_dir)[-2:]) + "first_15"
        dataset_tuple = (np.asarray(dataset_first_15),)
        dataset_first_15 = np.concatenate(dataset_tuple, axis=0)
        export_class_dataset(dataset_first_15, class_dir, name)

    return np.asarray(dataset)


def traffic_class_converter(dir_path):
    dataset_tuple = ()
    for file_path in [os.path.join(dir_path, fn) for fn in next(os.walk(dir_path))[2] if
                      (".csv" in os.path.splitext(fn)[-1])]:
        print("working on:", file_path)
        dataset_tuple += (traffic_csv_converter(file_path),)
    return np.concatenate(dataset_tuple, axis=0)


def iterate_all_classes():
    for class_dir in glob.glob(CLASSES_DIR):
        print("working on " + class_dir)
        dataset = traffic_class_converter(class_dir)
        print(dataset.shape)
        export_class_dataset(dataset, class_dir)


def random_sampling_dataset(input_array, size=2000):
    print("Import dataset " + input_array)
    dataset = np.load(input_array)
    print(dataset.shape)
    p = size * 1.0 / len(dataset)
    print(p)
    if p >= 1:
        raise Exception

    mask = np.random.choice([True, False], len(dataset), p=[p, 1 - p])
    dataset = dataset[mask]
    print("Start export dataset")

    np.save(os.path.splitext(input_array)[0] + "_samp", dataset)


if __name__ == '__main__':
    iterate_all_classes()
