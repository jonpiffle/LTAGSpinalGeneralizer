from collections import deque, defaultdict, Counter
from ltag_spinal import SpinalLTAG
from spinal_loader import UncompressedSpinalLTAGLoader
import matplotlib.pyplot as plt
import numpy as np
import os, pickle, json

def unlexicalized_spine(tree):
    return "(" + tree.label() + " " + " ".join([unlexicalized_spine(c) for c in tree if isinstance(c, SpinalLTAG)]) + ")"

def get_all_rules(tree):
    rules = []
    queue = deque([tree])
    while len(queue) > 0:
        cur_tree = queue.popleft()
        rules += cur_tree.rules
        queue += [c for c in cur_tree if isinstance(c, SpinalLTAG)]
    return rules

def generalized_tree_representation(tree):
    rules = get_all_rules(tree)
    return (unlexicalized_spine(tree), tree.tree_type, tree.roleset_id, tuple([generalized_rule_representation(r) for r in rules]))

def generalized_rule_representation(rule):
    return (rule.rule_type, rule.pos, rule.semantic_role, rule.action_location.treeposition, rule.action_location.slot)

def partition_trees(trees, partition_func=generalized_tree_representation):
    tree_dict = defaultdict(list)
    for tree in trees:
        tree_dict[partition_func(tree)].append(tree)
    return tree_dict

tree_loader = UncompressedSpinalLTAGLoader(filename="uncompressed_trees.json")
trees = tree_loader.load(limit=1000)

grouped = partition_trees(trees)

id_map = {}
for i, (tree_str, group) in enumerate(grouped.items()):
    for tree in group:
        id_map[tree.tree_id] = i
id_map[None] = None

attach_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for t in trees:
    attach_counts[str(id_map[t.parent_id])][str(t.parent_attach_id)][t.label()] += 1

# write one unique tree to a json file for each group
unique_trees = []
for tree_str, group in grouped.items():
    u_tree = group[0]
    u_tree_id = id_map[u_tree.tree_id]
    lexicalization = Counter([t.leaves()[0].lower() for t in group])
    u_attach_counts = attach_counts[str(u_tree_id)]
    t_dict = {
        'spine': unlexicalized_spine(u_tree),
        'tree_id': u_tree_id,
        'lexicalization': dict(lexicalization),
        'attach_counts': dict(u_attach_counts),
        'tree_type': u_tree.tree_type,
        'predicate': u_tree.predicate,
        'roleset_id': u_tree.roleset_id,
        'num_args': u_tree.num_args,
        'semantic_role': u_tree.semantic_role,
        'rules': [r.to_dict() for r in u_tree.rules],
    }
    unique_trees.append(t_dict)

with open('compressed_trees.json', 'w') as f:
   f.write(json.dumps(unique_trees))
