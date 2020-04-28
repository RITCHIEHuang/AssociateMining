#!/usr/bin/env python
# Created at 2020/4/26
import pprint
import time
from itertools import combinations
from math import ceil


def bf_frequent_items(df, items, item_counts, min_sup=0.05, debug=False):
    """
    generate all possible frequent item sets by relative min support
    >>> {1: {(('I5',), 2), (('I2',), 7), (('I1',), 6), (('I3',), 6), (('I4',), 2)},
         2: {(('I1', 'I2'), 4), (('I1', 'I3'), 4), (('I1', 'I5'), 2), (('I3', 'I2'), 4), (('I4', 'I2'), 2), (('I5', 'I2'), 2)},
         3: {(('I1', 'I3', 'I2'), 2), (('I1', 'I5', 'I2'), 2)}
         }
    :param df: dataframe
    :param items:
    :param item_counts: num of items
    :param min_sup: fractional relative min support
    :return:
    """
    print("Find frequent item sets by Brute Force")
    print("-" * 100)

    frequent_sets = {}  # dictionary, key-> k, value-> k item sets
    min_threshold = ceil(df.shape[0] * min_sup)

    for k in range(1, 1 + item_counts):
        k_item_subsets = combinations(items, k)  # all possible k-item sets
        time_start = time.time()
        # check satisfied k-item sets
        filtered_k_subsets = {(
            tuple(k_item_subset), (set(k_item_subset) <= df["items"]).sum()) for k_item_subset in k_item_subsets
            if
            (set(k_item_subset) <= df["items"]).sum() >= min_threshold}
        print(f"Process {k}-item subsets in {time.time() - time_start: .5f} s")
        # if k subsets support can't satisfy, k + 1, ... can't satisfy
        if len(filtered_k_subsets) <= 0:
            break
        frequent_sets[k] = filtered_k_subsets

    if debug:
        print("Final frequent item sets")
        print("=" * 100)
        pprint.pprint(frequent_sets)
        print("=" * 100)
    return frequent_sets
