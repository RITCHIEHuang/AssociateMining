#!/usr/bin/env python
# Created at 2020/4/26
import os
from functools import reduce

import numpy as np
import pandas as pd

GROCERY_STORE_DATA_PATH = "./dataset/GroceryStore/Groceries.csv"
UNIX_COMMANDS_DATA_PATH = "./dataset/UNIX_usage/"


def read_grocery_data(dataset_path=None):
    if dataset_path is None:
        df = pd.read_csv(GROCERY_STORE_DATA_PATH, index_col=0)
    else:
        df = pd.read_csv(dataset_path, index_col=0)
    db = pd.DataFrame()
    db["items"] = df["items"].apply(lambda x: set(x[1:-1].split(",")))
    items = reduce(lambda a, b: a | b, db.values)[0]
    item_counts = len(items)

    return db, items, item_counts


def read_unix_commands_data(dataset_path=None):
    SOF_PATTERN = "**SOF**"
    EOF_PATTERN = "**EOF**"

    if dataset_path is None:
        dataset_path = UNIX_COMMANDS_DATA_PATH

    # walk through all files and store items in pandas data frame
    df = pd.DataFrame()
    for root, dirs, files in os.walk(dataset_path):
        def read_file(file_path):
            print("Process file: ", file_path)
            cur_items = []
            if file_path:
                with open(file_path) as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line == SOF_PATTERN:
                            cur_item = set()
                        elif line == EOF_PATTERN:
                            if cur_item:
                                cur_items.append(cur_item)
                        else:
                            cur_item.add(line)
            return np.array(cur_items)

        file_abs_path_func = lambda x: os.path.join(root, x)
        file_data = np.c_[
            [read_file(path) for path in [file_abs_path_func(x) for x in files if x.startswith("sanitized")]]]
        if len(file_data):
            df = df.append(pd.DataFrame(file_data[0], columns=["items"]), ignore_index=True)
    items = reduce(lambda a, b: a | b, df.values)[0]
    item_counts = len(items)
    return df, items, item_counts


def read_dummy_data():
    """
    dummy data in text book <<DataMining. Concepts and Technique>> 3rd Edition, Chapter 6, Page 250
    :return:
    """
    df = pd.DataFrame({
        "TID": ["T100", "T200", "T300", "T400", "T500", "T600", "T700", "T800", "T900"],
        "List_of_item_ids": ["I1, I2, I5", "I2, I4", "I2, I3", "I1, I2, I4", "I1, I3", "I2, I3", "I1, I3",
                             "I1, I2, I3, I5", "I1, I2, I3"]
    })
    db = pd.DataFrame()
    db["transactions"] = df["List_of_item_ids"].apply(lambda x: set(x.split(", ")))
    items = reduce(lambda a, b: a | b, db.values)[0]
    item_counts = len(items)

    return db, items, item_counts


if __name__ == '__main__':
    # read_grocery_data(GROCERY_STORE_DATA_PATH)
    # read_dummy_data()
    read_unix_commands_data(UNIX_COMMANDS_DATA_PATH)
