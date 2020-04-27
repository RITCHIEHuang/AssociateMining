#!/usr/bin/env python
# Created at 2020/4/26

import pprint
from itertools import combinations
from math import ceil


def apriori_frequent_items(df, items, item_counts, min_sup=0.05):
    print("Find frequent item sets by Apriori algorithm")
    print("=" * 100)

    frequent_sets = {}
    min_threshold = ceil(df.shape[0] * min_sup)

    for k in range(1, 1 + item_counts):
        k_item_subsets = combinations(items, k)  # all possible k-item sets

        print(f"Process {k} subsets")
        # check satisfied k-item sets
        filtered_k_subsets = {(
            tuple(k_item_subset), (set(k_item_subset) <= df["items"]).sum()) for k_item_subset in k_item_subsets
            if
            (set(k_item_subset) <= df["items"]).sum() >= min_threshold}
        if len(filtered_k_subsets) <= 0:  # if k subsets support can't satisfy, k + 1, ... can't satisfy
            break
        frequent_sets[k] = filtered_k_subsets

    print("Final frequent item sets")
    print("=" * 100)
    pprint.pprint(frequent_sets)
    print("=" * 100)
    return frequent_sets
