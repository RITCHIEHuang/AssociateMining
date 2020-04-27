#!/usr/bin/env python
# Created at 2020/4/26
from functools import partial

from apriori.apriori import apriori_frequent_items
from brute_force.brute_force import bf_frequent_items
from data_process import read_grocery_data, read_dummy_data
from gen_strong_rule import generate_strong_rule

GROCERY_STORE_DATA_PATH = "./dataset/GroceryStore/Groceries.csv"

min_support = 0.21  # fractional min support
min_confidence = 0.5  # fractional min confidence

dummy_data_collect_func = read_dummy_data
grocery_data_collect_func = partial(read_grocery_data, dataset_path="./dataset/GroceryStore/Groceries.csv")
df, items, item_counts = dummy_data_collect_func()

###########################################
#  Brute force Baseline
###########################################

bf_frequent_item_sets = bf_frequent_items(df, items, item_counts, min_sup=min_support)
bf_strong_rules = generate_strong_rule(min_confidence, df, bf_frequent_item_sets)

###########################################
#  Apriori algorithm
###########################################
ap_frequent_item_sets = apriori_frequent_items(df, items, item_counts, min_sup=min_support)
ap_strong_rules = generate_strong_rule(min_confidence, df, ap_frequent_item_sets)

