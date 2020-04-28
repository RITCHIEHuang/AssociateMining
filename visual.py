#!/usr/bin/env python
# Created at 2020/4/28

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')

df = pd.read_csv('rules_time_cost.csv')

COL = 4
ROW = 5

fig, axes = plt.subplots(ROW, COL, figsize=(6 * COL, 4 * ROW))

for k in range(20):
    ax = axes[k // COL][k % COL]
    sns.lineplot(data=data, x=x_axis, y=y_axis, hue=hue, ci='sd', ax=ax, **kwargs)


