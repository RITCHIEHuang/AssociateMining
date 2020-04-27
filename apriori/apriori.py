#!/usr/bin/env python
# Created at 2020/4/26

import pprint
from itertools import combinations
from math import ceil


def apriori_frequent_items(df, items, item_counts, min_sup=0.05, debug=False):
    """
    >>>{1: [(('I1',), 6), (('I2',), 7), (('I3',), 6), (('I4',), 2), (('I5',), 2)],
        2: [(('I1', 'I2'), 4),(('I1', 'I3'), 4),(('I1', 'I5'), 2),(('I2', 'I3'), 4),(('I2', 'I4'), 2),(('I2', 'I5'), 2)],
        3: [(('I1', 'I2', 'I3'), 2), (('I1', 'I2', 'I5'), 2)]
        }
    generate frequent item sets by Apriori algorithm
    :param df:
    :param items:
    :param item_counts:
    :param min_sup:
    :param debug: debug mode
    :return:
    """
    print("Find frequent item sets by Apriori algorithm")
    print("=" * 100)

    frequent_sets = {}
    hash_sets = {}
    min_threshold = ceil(df.shape[0] * min_sup)

    # initialized by 1 frequent items
    # all elements sorted by dictionary order
    frequent_k_item_sets = sorted(
        ((tuple(item_set), (set(item_set) <= df["items"]).sum()) for item_set in combinations(items, 1)
         if (set(item_set) <= df["items"]).sum() >= min_threshold),
        key=lambda x: x[0])
    hash_k_sets = {item_set for item_set in combinations(items, 1) if
                   (set(item_set) <= df["items"]).sum() >= min_threshold}

    frequent_sets[1] = frequent_k_item_sets
    if debug:
        print("1-item frequent sets")
        pprint.pprint(frequent_k_item_sets)

    hash_sets[1] = hash_k_sets
    if debug:
        print("1-item hash sets")
        pprint.pprint(hash_k_sets)

    # perform level-wise generation by join two k - 1 frequent sets and pruning
    for k in range(2, 1 + item_counts):
        print(f"Process {k} subsets")
        cur_item_sets = []
        cur_hash_sets = set()
        for i in range(len(frequent_k_item_sets) - 1):
            for j in range(i + 1, len(frequent_k_item_sets)):
                # joining : find all candidate k item sets
                a, b = frequent_k_item_sets[i], frequent_k_item_sets[j]
                if a[0][:-1] == b[0][:-1] and a[0][-1] < b[0][-1]:
                    candidate_item_set = a[0] + (b[0][-1],)
                    # pruning : checking all k - 1 item subsets of candidate
                    candidate_subsets = set(
                        map(lambda x: tuple(sorted(x)), combinations(set(candidate_item_set), k - 1)))
                    if not candidate_subsets - hash_k_sets:
                        candidate_sup = (set(candidate_item_set) <= df["items"]).sum()
                        if candidate_sup >= min_threshold:
                            cur_item_sets.append((candidate_item_set, candidate_sup))
                            cur_hash_sets.add(candidate_item_set)

        if len(cur_item_sets) <= 0:
            break

        if debug:
            print(f"{k}-item frequent item sets")
            print(cur_item_sets)

        frequent_sets[k] = cur_item_sets
        frequent_k_item_sets = cur_item_sets

        if debug:
            print(f"{k}-item hash sets")
            print(cur_hash_sets)

        hash_sets[k] = cur_hash_sets
        hash_k_sets = cur_hash_sets

    if debug:
        print("Final frequent item sets")
        print("=" * 100)
        pprint.pprint(frequent_sets)
        print("=" * 100)
    return frequent_sets
