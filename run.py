#!/usr/bin/env python
# Created at 2020/4/26
import os
import time
from functools import partial, reduce

import numpy as np
import pandas as pd

from apriori.apriori import apriori_frequent_items
from brute_force.brute_force import bf_frequent_items
from data_process import read_grocery_data, read_dummy_data, read_unix_commands_data
from gen_strong_rule import generate_strong_rule

GROCERY_STORE_DATA_PATH = "./dataset/GroceryStore/Groceries.csv"
UNIX_COMMAND_DATA_PATH = "./dataset/Unix_usage/"


def test(output_dir="./result/"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # test code on dummy data
    min_support = 0.21  # fractional min support
    min_confidence = 0.5  # fractional min confidence

    dummy_data_collect_func = read_dummy_data
    df, items, item_counts = dummy_data_collect_func()

    ###########################################
    #  Brute force Baseline
    ###########################################

    bf_frequent_item_sets = bf_frequent_items(df, items, item_counts, min_sup=min_support)
    write_frequent_item_set_to_file(bf_frequent_item_sets,
                                    file_path=f"{output_dir}bf/frequent_set/sup_{min_support}_conf_{min_confidence}.txt")
    bf_strong_rules = generate_strong_rule(min_confidence, df, bf_frequent_item_sets)
    write_rule_to_file(bf_strong_rules, file_path=f"{output_dir}bf/rule/sup_{min_support}_conf_{min_confidence}.txt")

    ###########################################
    #  Apriori algorithm
    ###########################################
    ap_frequent_item_sets = apriori_frequent_items(df, items, item_counts, min_sup=min_support)
    write_frequent_item_set_to_file(bf_frequent_item_sets,
                                    file_path=f"{output_dir}ap/frequent_set/sup_{min_support}_conf_{min_confidence}.txt")
    ap_strong_rules = generate_strong_rule(min_confidence, df, ap_frequent_item_sets)
    write_rule_to_file(ap_strong_rules, file_path=f"{output_dir}ap/rule/sup_{min_support}_conf_{min_confidence}.txt")

    ###########################################
    #  Fp-growth algorithm
    ###########################################


def write_frequent_item_set_to_file(frequent_set, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(file_path, "w+") as f:
        frequent_item_set_count = sum(map(lambda a: len(a), frequent_set.values()))
        f.write(f"Find {frequent_item_set_count} frequent item sets !!!" + os.linesep)

        str_func = lambda a, b: f'Including {len(b)} {a}-item frequent set' + os.linesep + (
            reduce(lambda l1, l2: l1 + l2, [", ".join(item[0]) + ", support: " + str(
                item[1]) + os.linesep for item in b]))

        f.writelines([str_func(k, v) + os.linesep for k, v in frequent_set.items()])


def write_rule_to_file(rule_set, file_path):
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    with open(file_path, "w+") as f:
        f.write(f"Find {len(rule_set)} rules !!!" + os.linesep)
        rule_str_func = lambda rule: rule[0][0] + "===>" + rule[1][0][0] + ", confidence: " + str(
            rule[1][1]) + os.linesep
        f.writelines([rule_str_func(rule) for rule in rule_set])


# bench marks
def main(output_dir="./result/"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    """parameters"""
    min_sups = np.linspace(0.0001, 0.05, 20)
    min_confs = np.linspace(0.2, 0.8, 20)

    """data sets"""
    grocery_data_collect_func = partial(read_grocery_data, dataset_path=GROCERY_STORE_DATA_PATH)
    unix_commands_data_collect_func = partial(read_unix_commands_data, dataset_path=UNIX_COMMAND_DATA_PATH)
    data_funcs = [('grocery', grocery_data_collect_func), ('unix_usage', unix_commands_data_collect_func)]

    record_df = pd.DataFrame(
        columns=['min_sup', 'min_conf', 'data_set', 'num_items', 'num_transactions', 'brute_force', 'apriori',
                 'fp_growth'])

    experiment_id = 0
    for data_set, data_func in data_funcs:
        df, items, item_counts = data_func()
        for min_sup in min_sups:
            for min_conf in min_confs:
                print("=" * 150)
                print(f"Experiment {experiment_id + 1} setting: min support={min_sup}, min confidence={min_conf}")
                print(
                    f"Data set descriptions: data set: {data_set}, number of items: {item_counts}, number of transactions:"
                    f" {df.shape[0]}")
                print()

                ###########################################
                #  Brute force Baseline
                ###########################################
                time_start = time.time()
                bf_frequent_item_sets = bf_frequent_items(df, items, item_counts, min_sup=min_sup)
                bf_time_cost = time.time() - time_start
                print(f"Brute force spent {bf_time_cost} s for mining frequent item sets.")
                write_frequent_item_set_to_file(bf_frequent_item_sets,
                                                file_path=f"{output_dir}bf/frequent_set/sup_{min_sup}_conf_{min_conf}.txt")

                bf_strong_rules = generate_strong_rule(min_conf, df, bf_frequent_item_sets)
                write_rule_to_file(bf_strong_rules,
                                   file_path=f"{output_dir}bf/rule/sup_{min_sup}_conf_{min_conf}.txt")

                ###########################################
                #  Apriori algorithm
                ###########################################
                time_start = time.time()
                ap_frequent_item_sets = apriori_frequent_items(df, items, item_counts, min_sup=min_sup)
                ap_time_cost = time.time() - time_start
                print(f"Apriori spent {ap_time_cost} s for mining frequent item sets.")
                write_frequent_item_set_to_file(bf_frequent_item_sets,
                                                file_path=f"{output_dir}ap/frequent_set/sup_{min_sup}_conf_{min_conf}.txt")

                ap_strong_rules = generate_strong_rule(min_conf, df, ap_frequent_item_sets)
                write_rule_to_file(ap_strong_rules,
                                   file_path=f"{output_dir}ap/rule/sup_{min_sup}_conf_{min_conf}.txt")

                ###########################################
                #  TODO Fp-growth algorithm
                ###########################################

                record_df = record_df.append(pd.Series(
                    {"min_sup": min_sup, "min_conf": min_conf, "data_set": data_set, "num_items": item_counts,
                     "num_transactions": df.shape[0],
                     "brute_force": bf_time_cost, "apriori": ap_time_cost}), ignore_index=True)

                experiment_id = experiment_id + 1
                print("=" * 150)

    record_df.to_csv("record.csv")


if __name__ == '__main__':
    main(output_dir="./result/")