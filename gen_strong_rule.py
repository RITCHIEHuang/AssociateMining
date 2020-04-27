#!/usr/bin/env python
# Created at 2020/4/26
import pprint
from itertools import combinations


def generate_strong_rule(min_conf, df, frequent_sets):
    all_rules = set()
    # generate all rules A => B
    for k, vs in frequent_sets.items():
        if k <= 1:
            continue
        for v_sup in vs:
            *v, sup = v_sup
            v = v[0]
            # note that v is a k-item frequent set
            for t in range(1, k):
                frequent_subsets = list(combinations(v, t))  # non-empty set
                curr_rule = {A: (tuple(set(v) - set(A)), sup / (set(A) <= df[
                    "items"]).sum()) for A in frequent_subsets if
                             (sup / (set(A) <= df[
                                 "items"]).sum()) >= min_conf}  # complementary set of A
                # print(curr_rule)
                all_rules |= set(curr_rule.items())

    print("Final Strong rule")
    print("=" * 100)
    pprint.pprint(all_rules)
    print("=" * 100)

    return all_rules
