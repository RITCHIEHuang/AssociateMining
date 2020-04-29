#!/usr/bin/env python
# Created at 2020/4/28

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style='darkgrid')

df_alg = pd.read_csv('df_algorithm.csv')
df_rule = pd.read_csv('df_rule.csv')

COL = 4
ROW = 5

fig, axes = plt.subplots(ROW, COL, figsize=(5 * COL, 4 * ROW))
for idx, min_sup in enumerate(df_rule["min support"].unique()):
    sub_df = df_rule[df_rule["min support"] == min_sup]
    ax = axes[idx // COL][idx % COL]
    ax.set_title(f"min support = {min_sup:.3f}")
    sns.lineplot(data=sub_df, x="min confidence", y="rule nums", ci='sd', ax=ax)


plt.figure()
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71", "#e67e22", "#f1c40f"]
sns.set(palette=sns.color_palette(flatui))
sns.lineplot(x="min support", y="freq_set nums", ci='sd', data=df_alg)
# g = sns.FacetGrid(df_rule, col="min support", col_wrap=5)
# g.map(sns.lineplot, "min confidence", "rule nums")
# sns.lineplot(x="min confidence", y="rule nums", hue="min support", data=df_rule)
plt.show()
