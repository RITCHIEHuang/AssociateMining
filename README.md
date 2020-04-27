# 关联规则挖掘

从给定数据集中挖掘符合一定条件的关系(A -> B)

## 规则
规则的形式:
Body -> Head [Support, Confidence]

具体的定义，给定规则 A -> B (A 和 B 不交)
1. Support(A -> B) = Pr[A and B] 是联合概率
2. Confidence(A -> B) = Pr[B | A] 是一个条件概率

1. 强关联规则

满足 Support(A -> B) >= min_sup && Confidence(A -> B) >= min_conf

2. k 项集

包含 k 个 items 的集合

3. 频繁项集

满足 min_sup 的项集
 
## 规则挖掘方法

主要分为两部，其中第 2 步可以借助于第一步，因此，第 1 步是关键。

1. 找出所有频繁项集

- 暴力穷举
    
- Apriori

- FP-Growth

2. 从频繁项集中生成强关联规则

![强关联规则生成](https://tva1.sinaimg.cn/large/007S8ZIlgy1ge6ylznkegj31wg0kemyj.jpg)