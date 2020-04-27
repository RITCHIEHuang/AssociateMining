#!/usr/bin/env python
# Created at 2020/4/27
import math
from data_process import read_dummy_data


class TreeNode:
    def __init__(self, item_value, item_count, parent_node):
        self.item_name = item_value
        self.count = item_count
        self.node_link = None
        self.parent = parent_node
        self.children = {}


def create_set(data_set):
    # 转换数据格式{frozenset:count}->dict
    ret_dic = {}
    for trans in data_set:
        ret_dic[frozenset(trans)] = ret_dic.get(frozenset(trans), 0) + 1
    return ret_dic


def update_header_table(pointer: TreeNode, new_node: TreeNode):
    # 更新指向fp-tree的头表指针
    # cur_pointer = copy.deepcopy(pointer)
    # while cur_pointer.node_link is not None:
    #     cur_pointer = cur_pointer.node_link
    # cur_pointer.node_link = new_node
    while pointer.node_link is not None:
        pointer = pointer.node_link
    pointer.node_link = new_node


def update_tree(order_transaction, root_node, header_table, count):
    """
    基于一条排序的只包含频繁项的transaction来更新fp-tree
    :param order_transaction:
    :param root_node:
    :param header_table:
    :param count: the number of the same transaction
    :return:
    """
    for item in order_transaction:
        current_child = root_node.children
        if item in current_child.keys():
            current_child[item].count += count
            root_node = current_child[item]
        else:
            current_child[item] = TreeNode(item, count, parent_node=root_node)
            if header_table[item][1] is None:
                header_table[item][1] = current_child[item]
            else:
                update_header_table(header_table[item][1], current_child[item])
            root_node = current_child[item]


def create_tree(data_set, min_sup):
    """
    基于当前的transaction集合(Conditional pattern-base)构建fp-tree和header table
    :param data_set: transection set in the form like {(f,c,a,m}:2, {c,b}:1}
    :param min_sup:
    :return:
    """
    # scan current DB, find frequent 1-itemset
    header_table = {}
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]# header_table[item] + data_set[trans]
    for item_1 in list(header_table.keys()):
        if header_table[item_1] < min_sup:
            del (header_table[item_1])
    fre_item_set = set(header_table.keys())
    if len(fre_item_set) == 0:
        return None

    # initialize header table and fp-tree
    for item_1 in header_table:
        # header table item includes name and node count and address of node:{item_name:[frequency, pointer]}
        header_table[item_1] = [header_table[item_1], None]
    ret_tree = TreeNode("root node", 1, parent_node=None)

    # scan current DB again, construct fp-tree
    for transaction, count in data_set.items():
        local_id = {}
        for item_1 in transaction:
            if item_1 in fre_item_set:
                local_id[item_1] = header_table[item_1][0]
        if len(local_id) > 0:
            # ordered transaction that removes infrequent items
            order_set = [v[0] for v in sorted(local_id.items(), key=lambda p: p[1],
                                              reverse=True)]
            update_tree(order_set, ret_tree, header_table, count)
    return header_table


def before_tree(header_table_node, bef_path):
    if header_table_node.parent is not None:
        bef_path.append(header_table_node.item_name)
        before_tree(header_table_node.parent, bef_path)


def find_path(tree_node):
    # 找到当前树结点的Conditional pattern-base
    cond_pats = {}
    while tree_node is not None:
        pre_path = []
        before_tree(tree_node, pre_path)
        if len(pre_path) > 1:
            cond_pats[frozenset(pre_path[1:])] = tree_node.count
        tree_node = tree_node.node_link
    return cond_pats


def mining_fp_tree(header_table, min_sup, pre_path, fre_item_set, fre_set_count):
    """
    基于当前的fp-tree挖掘频繁项集，并递归生成conditional pattern-tree
    :param header_table:
    :param min_sup:
    :param pre_path: previous path to get current conditional pattern-tree
    :param fre_item_set:
    :param fre_set_count:
    :return:
    """
    fre_items = [v[0] for v in sorted(header_table.items(), key=lambda p:p[1][0])]
    for base_pat in fre_items:
        # add new frequent set
        new_fre_set = pre_path.copy()
        new_fre_set.add(base_pat)
        fre_set_count[frozenset(new_fre_set)] = header_table[base_pat][0]
        fre_item_set.append(new_fre_set)

        # recursively generate new tree
        cond_pat_path = find_path(header_table[base_pat][1])
        my_header = create_tree(cond_pat_path, min_sup)

        # recursively mining new tree
        if my_header is not None:
            mining_fp_tree(my_header, min_sup, new_fre_set, fre_item_set, fre_set_count)
    return fre_item_set, fre_set_count


def fp_growth_frequent_items(df, min_sup=0.3):
    """
    基于fp-growth算法生成频繁项集
    :param df: transaction set -> dataframe
    :param min_sup:
    :return:
    """
    min_threshold = math.ceil(df.shape[0] * min_sup)

    # build the first fp-tree and header table
    initial_data_set = [list(transaction) for transaction in df['items']]
    formed_data_set = create_set(initial_data_set)
    header_table = create_tree(formed_data_set, min_threshold)

    # recursively do the following step: caculate Conditional pattern-base and Conditional pattern-tree
    fre_item_set = []
    fre_item_count = {}
    mining_fp_tree(header_table, min_threshold, set([]), fre_item_set, fre_item_count )
    return fre_item_set, fre_item_count


if __name__ == '__main__':
    df, * _ = read_dummy_data()
    frequent_item_set, frequent_set_count = fp_growth_frequent_items(df=df, min_sup=0.21)
    print(frequent_set_count)
    print(frequent_item_set)
