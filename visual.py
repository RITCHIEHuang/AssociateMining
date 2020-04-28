#!/usr/bin/env python
# Created at 2020/4/28

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style='darkgrid')

df = pd.read_csv('rules_time_cost.csv')

COL = 4
ROW = 5

fig, axes = plt.subplots(ROW, COL, figsize=(6 * COL, 4 * ROW))
for idx, min_sup in enumerate(df["min support"].unique()):
    sub_df = df[df["min support"] == min_sup]
    ax = axes[idx // COL][idx % COL]
    ax.set_title(f"Min support = {min_sup:.3f}")
    sns.lineplot(data=sub_df, x="min confidence", y="running time", ci='sd', ax=ax)


# sns.lineplot(x="min confidence", y="running time", hue="min support", data=df)
plt.show()
