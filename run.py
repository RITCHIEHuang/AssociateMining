#!/usr/bin/env python
# Created at 2020/4/26
import os
import time
from functools import partial, reduce

import pandas as pd

from apriori.apriori import apriori_frequent_items
from brute_force.brute_force import bf_frequent_items
from data_process import read_grocery_data, read_dummy_data, read_unix_commands_data
from fp_growth.fp_growth import fp_growth_frequent_items
from gen_strong_rule import generate_strong_rule

GROCERY_STORE_DATA_PATH = "./dataset/GroceryStore/Groceries.csv"
UNIX_COMMAND_DATA_PATH = "./dataset/UNIX_usage/"


def test(output_dir="./result/", debug=False):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # test code on dummy data
    min_sup = 0.21  # fractional min support
    min_conf = 0.5  # fractional min confidence

    dummy_data_collect_func = read_dummy_data
    df, items, item_counts = dummy_data_collect_func()

    ###########################################
    #  Brute force Baseline
    ###########################################
    time_start = time.time()
    bf_frequent_item_sets = bf_frequent_items(df, items, item_counts, min_sup=min_sup, debug=debug)
    bf_time_cost = time.time() - time_start
    print(f"Brute force spent {bf_time_cost} s for mining frequent item sets.")
    write_frequent_item_set_to_file(bf_frequent_item_sets,
                                    file_path=f"{output_dir}bf/frequent_set/sup_{min_sup}.txt")
    bf_strong_rules = generate_strong_rule(min_conf, df, bf_frequent_item_sets, debug=debug)
    write_rule_to_file(bf_strong_rules, file_path=f"{output_dir}bf/rule/sup_{min_sup}_conf_{min_conf}.txt")

    ###########################################
    #  Apriori algorithm
    ###########################################
    time_start = time.time()
    ap_frequent_item_sets = apriori_frequent_items(df, items, item_counts, min_sup=min_sup, debug=debug)
    ap_time_cost = time.time() - time_start
    print(f"Apriori spent {ap_time_cost} s for mining frequent item sets.")
    write_frequent_item_set_to_file(ap_frequent_item_sets,
                                    file_path=f"{output_dir}ap/frequent_set/sup_{min_sup}.txt")
    ap_strong_rules = generate_strong_rule(min_conf, df, ap_frequent_item_sets, debug=debug)
    write_rule_to_file(ap_strong_rules, file_path=f"{output_dir}ap/rule/sup_{min_sup}_conf_{min_conf}.txt")

    ###########################################
    #  Fp-growth algorithm
    ###########################################
    time_start = time.time()
    *_, fp_frequent_item_sets = fp_growth_frequent_items(df=df, min_sup=min_sup, debug=debug)
    fp_time_cost = time.time() - time_start
    print(f"FP-growth spent {fp_time_cost} s for mining frequent item sets.")
    write_frequent_item_set_to_file(fp_frequent_item_sets,
                                    file_path=f"{output_dir}fp/frequent_set/sup_{min_sup}.txt")
    fp_strong_rules = generate_strong_rule(min_conf, df, fp_frequent_item_sets, debug=debug)
    write_rule_to_file(fp_strong_rules,
                       file_path=f"{output_dir}fp/rule/sup_{min_sup}_conf_{min_conf}.txt")


def write_frequent_item_set_to_file(frequent_set, file_path):
    """
    write frequent item sets to file
    :param frequent_set:
    :param file_path:
    :return:
    """
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(file_path, "w+") as f:
        frequent_item_set_count = sum(map(lambda a: len(a), frequent_set.values()))
        f.write(f"Find {frequent_item_set_count} frequent item sets !!!" + os.linesep)

        str_func = lambda a, b: f'Including {len(b)} {a}-item frequent set' + os.linesep + (
            reduce(lambda l1, l2: l1 + l2, [", ".join(item[0]) + ", support: " + str(
                item[1]) + os.linesep for item in b]))

        f.writelines([str_func(k, v) + os.linesep for k, v in frequent_set.items()])


def write_rule_to_file(rule_set, file_path):
    """
    write rules to file
    :param rule_set:
    :param file_path:
    :return:
    """
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_path, "w+") as f:
        f.write(f"Find {len(rule_set)} rules !!!" + os.linesep)
        rule_str_func = lambda rule: ", ".join(rule[0]) + "===>" + ", ".join(rule[1][0]) + ", confidence: " + str(
            rule[1][1]) + os.linesep
        f.writelines([rule_str_func(rule) for rule in rule_set])


# bench marks
def main(output_dir="./result/"):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    """parameters"""
    # min_sups = np.linspace(0.0001, 0.05, 20)
    # min_confs = np.linspace(0.2, 0.8, 20)

    min_sups = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    min_confs = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

    """data sets"""
    grocery_data_collect_func = partial(read_grocery_data, dataset_path=GROCERY_STORE_DATA_PATH)
    unix_commands_data_collect_func = partial(read_unix_commands_data, dataset_path=UNIX_COMMAND_DATA_PATH)
    data_funcs = [('grocery', grocery_data_collect_func), ('unix_usage', unix_commands_data_collect_func)]

    record_df = pd.DataFrame(
        columns=['min_sup', 'min_conf', 'data_set', 'num_items', 'num_transactions', 'algo', 'time'])

    experiment_id = 0
    for data_set, data_func in data_funcs:
        df, items, item_counts = data_func()
        for min_sup in min_sups:
            ###########################################
            #  Brute force Baseline
            ###########################################
            if min_sup >= 0.05:
                time_start = time.time()
                bf_frequent_item_sets = bf_frequent_items(df, items, item_counts, min_sup=min_sup)
                bf_time_cost = time.time() - time_start
                print(f"Brute force spent {bf_time_cost} s for mining frequent item sets.")
                write_frequent_item_set_to_file(bf_frequent_item_sets,
                                                file_path=f"{output_dir}bf/frequent_set/sup_{min_sup}.txt")

                for min_conf in min_confs:
                    print("=" * 150)
                    print(
                        f"Experiment {experiment_id + 1} setting: Algo: Brute force, min support: {min_sup}, min confidence: {min_conf}")
                    print(
                        f"Data set descriptions: data set: {data_set}, number of items: {item_counts}, number of transactions:"
                        f" {df.shape[0]}")
                    print()
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
            write_frequent_item_set_to_file(ap_frequent_item_sets,
                                            file_path=f"{output_dir}ap/frequent_set/{data_set}_sup_{min_sup}.txt")

            for min_conf in min_confs:
                print("=" * 150)
                print(
                    f"Experiment {experiment_id + 1} setting: Algo: Apriori, min support: {min_sup}, min confidence: {min_conf}")
                print(
                    f"Data set descriptions: data set: {data_set}, number of items: {item_counts}, number of transactions:"
                    f" {df.shape[0]}")
                print()

                ap_strong_rules = generate_strong_rule(min_conf, df, ap_frequent_item_sets)
                write_rule_to_file(ap_strong_rules,
                                   file_path=f"{output_dir}ap/rule/{data_set}_sup_{min_sup}_conf_{min_conf}.txt")

                record_df = record_df.append(pd.Series(
                    {"min_sup": min_sup, "min_conf": min_conf, "data_set": data_set, "num_items": item_counts,
                     "num_transactions": df.shape[0],
                     "algo": "Apriori",
                     "time": ap_time_cost}),
                    ignore_index=True)
                experiment_id = experiment_id + 1
                print("=" * 150)

            ###########################################
            #  Fp-growth algorithm
            ###########################################
            time_start = time.time()
            *_, fp_frequent_item_sets = fp_growth_frequent_items(df=df, min_sup=min_sup)
            fp_time_cost = time.time() - time_start
            print(f"Fp-growth spent {fp_time_cost} s for mining frequent item sets.")
            write_frequent_item_set_to_file(fp_frequent_item_sets,
                                            file_path=f"{output_dir}fp/frequent_set/sup_{min_sup}.txt")
            for min_conf in min_confs:
                print("=" * 150)
                print(
                    f"Experiment {experiment_id + 1} setting: Algo: FP-growth, min support: {min_sup}, min confidence: {min_conf}")
                print(
                    f"Data set descriptions: data set: {data_set}, number of items: {item_counts}, number of transactions:"
                    f" {df.shape[0]}")
                print()
                fp_strong_rules = generate_strong_rule(min_conf, df, fp_frequent_item_sets)
                write_rule_to_file(fp_strong_rules,
                                   file_path=f"{output_dir}fp/rule/sup_{min_sup}_conf_{min_conf}.txt")

    record_df.to_csv("record.csv")


if __name__ == '__main__':
    """
    流程:
    
    1. 指定 min support 用某种算法 (brute force, apriori, fp-growth) 挖掘出频繁项集，并写入文件
    2. 根频繁项集生成满足 min confidence 的强规则， 并写入文件
    
    example:
    >>>
    *_, fp_frequent_item_sets = fp_growth_frequent_items(df=df, min_sup=min_support)
    write_frequent_item_set_to_file(fp_frequent_item_sets,
                                    file_path=f"{output_dir}fp/frequent_set/sup_{min_support}_conf_{min_confidence}.txt")
    fp_strong_rules = generate_strong_rule(min_confidence, df, fp_frequent_item_sets)
    write_rule_to_file(fp_strong_rules,
                       file_path=f"{output_dir}fp/rule/sup_{min_support}_conf_{min_confidence}.txt")

    """
    main(output_dir="./result/")
