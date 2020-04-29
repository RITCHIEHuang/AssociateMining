# 关联规则挖掘

## 1. 问题及说明

​	    关联规则挖掘方法用于从数据集中挖掘出用户感兴趣的数据项之间的联系。关联规则挖掘受到商场购物篮分析的启发，通过发掘用户放置在购物篮中的商品之间的联系来分析用户的购买习惯，合理的关联规则可以应用于商品推荐和促销策略。

​        本次实验主要使用 Python 实现了 Brute force、Apriori 和 FP-growth 算法，来比较它们在空间占用和时间开销上的差异，实验使用的数据集包括 GroceryStore 的购买记录和UNIX 命令行使用记录。GroceryStore 数据集包含 9835 条交易记录，总共涉及 169 种不同的商品项，UNIX 命令行数据集收集了经过整理的8名用户的命令行使用记录，经过统计，该数据集包含9100条Session记录，包含2357种不同的命令项。

## 2. 算法和实现

​		 给定最小支持度min_sup 和最小置信度min_conf, 这里使用相对支持度。关联规则挖掘方法一般被拆分成两部分：

（1）查找频繁项集： 从所有的项集中找到出现频率大于min_sup的项集；

（2）从频繁项集中生成强关联规则：根据定义这些规则必须满足min_sup 和 min_conf 的要求。

​        由于第（2）步可以在（1）的基础上完成，因此，关联规则挖掘算法主要关注于（1）的实现，即高效地筛选出所有的频繁项集。

### 2.1  频繁项集挖掘算法

​        为了方便地分析算法的效率，假定数据集共有 $M$ 条记录，$N$ 个数据项。

#### 2.1.1 Brute force

​	    为了对比数据挖掘算法的效率，暴力算法 Brute force 这里仅作为 Baseline 实现。暴力算法的思路很简单：遍历所有可能的项集组合，统计每个项集的支持度，筛选出支持度不小于min_sup的项集。

​        代码实现上，通过 combination() 遍历数据项的组合，然后根据min_sup过滤掉不满足要求的项集，分层存储到字典中。这里的实现中，对算法做了剪枝，当 $k$ 项集无法满足要求时，必然有 $k + 1$项集无法满足要求，因此，算法可以结束。

​		代码实现见[brute_force.py](brute_force/brute_force.py)。

​	    暴力算法最大的缺陷在于其低下的效率，考虑生成所有可能的项集的所有可能情况共有 $\sum_{k = 1}^N \binom{N}{k}$ 种，而每次筛选都需要遍历数据集。

#### 2.1.2 Apriori

​		Apriori 算法的改进在于它利用了频繁集的“先验知识”，它按照递增的层序来生成项集，从 $k$ 项集中生成 $k+1$ 项集。

​        Apriori 算法利用了前面提到的暴力算法的一个剪枝技巧的逆向思路：任何频繁子集的非空子集都是频繁子集。根据这一特性，Apriori 算法通过两种操作实现了频繁子集生成：

（1）扩展操作：用两个 $k$ 项频繁集 $l_1, l_2$ 生成候选 $k+1$ 项集，算法使用了字典序排序的技巧，因此通过让两个 $k$ 项集的前 $k - 1$ 个元素都相同，而第 $k$ 个元素严格满足 $l_1 [k - 1] < l_2 [k -1]$，则可以保证不重不漏地生成满足条件的候选 $k+1$ 项集。

（2）剪枝操作：对扩展操作生成的候选 $k+1$ 项集，利用 Apriori 性质，检查所有的 $k$
项子集，若存在非频繁项集，则该候选集非频繁集。

​        代码实现见[apriori.py](apriori/apriori.py)，首先生成初始 $1-item$ 频繁项集，然后按照层级迭代式地进行扩展操作得到下一层的候选集，同时维护一个当前的频繁项哈希集，然后对候选集的元素进行子集验证，删除子集非频繁的候选项。当前层级已经没有满足要求的频繁项集时，迭代结束。		


#### 2.1.3 FP-growth

​     	Apriori 算法相较于暴力算法已经有较大的提升，但仍存在两个比较严重的瓶颈：

（1）当整个数据集的记录较多时，Apriori 算法通过层级扩展的操作仍会引入较大数量集的候选子集生成，例如整个数据集存在 $10^4$ 个数据项的时候，仍可能需要生成超过 $10^7$ 数量级的 $2-item$ 候选集；

（2）虽然已经引入了剪枝操作，Apriori 算法仍需要对剩下的候选集进行数据集的扫描来检查是否满足 min_sup 的要求。

​        为了解决这两个问题，FP-growth 算法出现了。它利用分治策略，实现了一种叫 FP-tree的数据结构，然后基于该数据结构进行频繁项集的构造，从而有效地避免了大量的数据集扫描。

​         FP-growth算法流程如下:

> 1）首先对整个数据集进行一次扫描，并根据 support 降序排列的 $1-item$ 项集；
>
> 2）然后构造 FP-tree，以 null 节点为树根，对整个数据集进行第二次扫描，并对每条数据项构造分支。

​	    代码实现见 [fp_growth.py](fp_growth/fp_growth.py)

### 2.2 强关联规则生成

​		基于已经挖掘出的频繁项集，可以生成满足 min_conf 的强关联规则。对于任何一条强关联规则 $A \Rightarrow B$，其置信度可以根据如下公式进行计算:

$$ confidence(A \Rightarrow B) = P(B | A) = \frac{sup\_count(A \cap B)}{sup\_count(A)}$$	

​	    强关联规则的条件概率中的表达式与 support_count 有关，因此只需查询对应的数据项即可。强关联规则的生成算法如下：

（1）对任意频繁项集 $l$，生成其非空子集；

（2）对于任意 $l$ 的非空子集 $s$，检查置信度条件  $\frac{sup\_count(l)}{sup\_count(s)} \geq min\_conf $ ，若满足，则生成对应的一条规则  $s \Rightarrow (l - s)$。

​        代码实现见 [gen_strong_rule.py](gen_strong_rule.py)


## 3 数据组织和实验方法

### 3.1 数据处理

​	    为了方便后续的算法处理，首先将两个数据集以相同的格式处理，以确保算法实现能够在两个数据集上进行挖掘。由于算法可能需要对数据集进行扫描和查找，因此使用基于集合的数据结构进行保存会极大地方便之后的算法实现。

#### 3.1.1 GroceryStore 数据集

​		数据集直接给出了 `Grocerie s.csv` 文件，该数据集共有 `9835` 条数据记录，包括 `169` 个不同的数据项。实现中，为了方便查找数据集，将每条记录转换成为一个数据项的集合 (set)，格式如下:

```
{'citrus fruit', 'semi-finished bread', 'margarine', 'ready soups'}
{'tropical fruit', 'yogurt', 'coffee'}
{'whole milk'}
```

#### 3.1.2 UNIX_usage 数据集

​		数据集给出了 9 个文件夹和数据，同样地将所有数据文件合并至一个单独的`csv`文件中以方便之后的算法处理，处理后的数据格式如下:

```
{'ls', 'source', '<1>', 'cd'}
{'&', '<1>', 'emacs', 'source', 'netscape', 'quota'}
{'emacs'}
{'&', 'netscape'}
{'&', 'emacs', 'netscape', 'quota'}
{'&', '<1>', 'finger', 'more', 'rm', 'netscape'}
```

### 3.2 实验方法

#### 3.2.1 实现顺序

​		根据现有的理论和认知，可以知道三种方法的效率大概是: Brute force < Apriori < FP-growth，因此，从实现上，通常考虑由简单到复杂的过程，这样更利于分析算法的本质。

#### 3.2.2 算法正确性检验

​		为了方便检验算法实现是否正确，这里使用了教材上的一个小型数据集 `AllElectronics`。它包含如下的数据内容:

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1ge9t65dlhlj309j07zjr8.jpg)

​	    数据集很小，三种算法都能很快运行完成可以用于检验算法的正确性。首先在这个小数据集上挖掘出正确的结果，以检验算法在实现上是正确的。

#### 3.2.3 算法效率对比方案

​	    `GroceryStore` 和 `UNIX_usage` 两个数据集在规模上的差异、算法效率的差异以及计算机本身性能的限制，同时也为了尽可能地体现算法的效率，制定的方案如下:

- 频繁集生成算法对比

  > 1. 对 Brute force 和 Apriori  两种算法分层（Level wise）进行时间对比实验（FP-growth 算法是不分层的），因为 Brute force 在高层次上无法在短期内运行出结果；
  > 2. 对 Apriori 和 FP-growth 算法在计算频繁集总体运行时间上进行对比实验，这样就可以间接地对比三种算法的效率；
  > 3. 对 `GroceryStore` 和 `UNIX_usage` 两个数据集采用不同的参数（min support），这直接与数据集相关，同时，由于 `UNIX_usage` 数据集较大，因此在该数据集上进行较少的对比实验。

- 强规则生成算法对比

  > 选择多组 min support，每次固定 min support 后，选择不同的 min confidence 进行实验，可以得到强规则生成算法的效率随 confidence 的变化规律。

- 时间和空间效率对比

  > 1. 时间效率对比：可以直接打印算法执行的时间；
  >
  > 2. 空间效率对比：由于使用 Python 进行算法实现，生成器表达式的使用，无法直接体现算法的空间需求，因此，通过查看系统内存占用和理论分析进行对比。

## 4 实验结果及分析

### 4.1 GroceryStore 数据集

#### 4.1.1 实例

- 实验参数设置：

```
min_support: 0.01
min_confidence: 0.3
```

- 时间对比：

| 算法        | 1 频繁项集 | 2 频繁项集 | 3 频繁项集 | 4 频繁项集 | 总时间|
| ----------- | ---------- | ---------- | ---------- | ---------- | --------|
| Brute force | 0.40176 s  | 18.16829 s | 893.30984 s | —— | 无法有效统计 |
| Apriori     | 0.33339 s  | 6.02452 s  | 1.12626 s  | 0.00791 s  | 7.49208 s |
| FP-growth   | —— |——|——|——|0.65571 s|

- 生成的频繁项集（部分）结果：

```
Find 333 frequent item sets !!!
Including 88 1-item frequent set
UHT-milk, support: 329
baking powder, support: 174
beef, support: 516
berries, support: 327
beverages, support: 256
bottled beer, support: 792
bottled water, support: 1087
brown bread, support: 638
butter, support: 545
butter milk, support: 275
cake bar, support: 130
candy, support: 294
......
```

- 生成的强关联规则（部分）结果：

```
Find 125 rules !!!
whipped/sour cream, yogurt===>whole milk, confidence: 0.5245098039215687
tropical fruit, whole milk===>other vegetables, confidence: 0.40384615384615385
butter milk===>whole milk, confidence: 0.41454545454545455
beef===>root vegetables, confidence: 0.3313953488372093
other vegetables, pork===>whole milk, confidence: 0.4694835680751174
onions===>other vegetables, confidence: 0.45901639344262296
white bread===>whole milk, confidence: 0.4057971014492754
pork===>whole milk, confidence: 0.3844797178130511
other vegetables, tropical fruit===>whole milk, confidence: 0.47592067988668557
rolls/buns, yogurt===>other vegetables, confidence: 0.3343195266272189
cream cheese ===>whole milk, confidence: 0.4153846153846154
sliced cheese===>whole milk, confidence: 0.43983402489626555
rolls/buns, root vegetables===>other vegetables, confidence: 0.502092050209205
....
```
​		可以看到一条规则 `whipped/sour cream, yogurt===>whole milk, confidence: 0.5245098039215687` ，这表明当用户购买生奶油/酸奶油和酸奶时，有`0.525` 置信度购买全脂牛奶，可以发现这几样商品是制作冰淇淋的原料，所以人们可以根据这种习惯将生奶油和纯牛奶放在一起促销，这也就显示了关联规则在实际生活中的作用。
#### 4.1.2 频繁集挖掘时间

​        实验选择了 20 组不同的 min_support，并对比 Apriori 算法和 FP-growth 算法在时间效率上的差异，由于暴力算法是在太慢了，这里就不做对比了。

​	    根据实验方案，对 Apriori 算法和 FP-growth 算法的时间效率进行对比，选择了20组 min support 进行对照实验，可以看到 Apriori 算法在 min support 较小时效率明显低于 FP-growth。

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1geadqkuopkj30ye0piglt.jpg)

#### 4.1.3 频繁集数量

​		实验共选择了 20 组不同的 min support，随着 min support 增加，生成的频繁项集的数量逐渐减少，其数量变化趋势如下图:

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1geado6tajvj30ys0pi74h.jpg)

#### 4.1.4 强规则数量

 	   实验共进行了 20 组，这里只给出了 6 组，这里反映了在给定的 min support 下，随着 min confidence 的增大规则数量的减少。

![](https://tva1.sinaimg.cn/large/007S8ZIlgy1geadmsohhhj31ko0u0wi1.jpg)

### 4.2 UNIX_usage 数据集

- 实验参数设置:

```
min_support: 0.01
min_confidence: 0.2
```

- 时间对比:

| 算法        | 1 频繁项集 | 2 频繁项集 | 3 频繁项集 | 4 频繁项集 | 5 频繁项集 | 6 频繁项集 | 7 频繁项集 | 8 频繁项集 | 9频繁项集 | 总时间 |
| ----------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---------- | ----- | ----- |
| Brute force | 3.05911 s | 3571.29526 s | —— | —— | —— | —— | —— | —— | —— | 太慢了 |
| Apriori     | 3.24454 s | 6.97719 s | 5.12690 s | 4.72359 s | 4.06608 s | 1.97292 s | 0.23978 s | 0.00599 s | 0.00001 s | 29.36919 s |
| FP-growth   | —— | —— | —— | —— | —— | —— | —— | —— | —— |1.25494 s|

   在该数据集上的分析方法与 GroceryStore 数据集相同，但是由于时间限制，没有能够完成更多的实验。

## 4.3 内存占用对比

​        实验中，内存占用在暴力算法上尤为突出，但是由于使用 Python 语言的关系，使用生成器表达式可以减少内存消耗，从计算机的内存监视器没有观察到明显的变化。

### 4.3.1 Brute force

​		该算法的内存占用在实现中主要是存储可能的 $k$ 项集的组合，假定共有 $M$ 条数据记录，$N$ 个数据项，则枚举组合的空间占用为: $\sum_{k = 1}^{N} \binom{N}{k}$，而实际实现中会采用迭代式的生成方法，并进行剪枝，则其存储开销不会超过 $\max_k \binom{N}{k} $。

###  4.3.2 Apriori

​		该算法可以考虑对 Brute force 的优化，每次合并两个 $k$ 项集得到 $k + 1$ 项集，因此，可以保证第 $k+1$ 层的内存占用不超过 $k$ 层内存占用的平方，但本质上还是在枚举组合。

###  4.3.3 FP-growth

​		该算法的存储结构决定了：其空间占用不超过: $N * M$，而考虑到公共前缀的优化，应该会远远小于 $N * M $ ，其生成过程充分利用了 FP-tree 本身的前缀性质，避免了大规模子集的枚举。

## 5 总结

​        使用 Python 实现了几种频繁项集的关联挖掘算法，设计了比较完整和合理的实验设置，并在 `GroceryStore` 上进行了较完整的时间效率比较，在 `UNIX_usage` 数据集的实验不够完整，对于内存占用效率的分析也欠缺很多。

​		暴力算法由于需要遍历所有的组合，会面临指数爆炸，该方法不可用；Apriori 算法仍面临大数据集上的频繁扫描，在大规模数据集上无法胜任，FP-growth 则是更高效的挖掘算法，但实现上会比 Apriori 复杂，这也就是一种舍与得的权衡。

​		另外，这也是对于几种算法在实际数据集上的应用，Apriori 和 FP-growth 都是十分优秀的数据挖掘算法，它们确实能对解决实际问题产生巨大的帮助，同时，也是由于实际问题的规模不断扩大，问题越来越复杂，也对数据挖掘算法的设计和优化提出了更高的要求。


