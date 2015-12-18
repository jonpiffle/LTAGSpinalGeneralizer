import json, re, os, pickle
from itertools import tee
from spinal.ltag_spinal import SpinalLTAG, Rule

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class SpinalLTAGLoader(object):
    """Class to load a treebank in the generalized tree format that was converted from Libin Shen's treebank """

    def __init__(self, filename="trees.json"):
        self.filename = filename

    def load(self, limit=None):
        raise NotImplementedError

    def add_rule_to_tree(self, root, rule):    
        """ 
        Given the root of a tree and a rule, attach the rule to the node it is supposed to act on
        and update the rules position to (), meaning self

        If this is not possible (for dependencies on locations not added to the tree yet), the rule remains at the root
        """

        try:
            root[rule.action_location.treeposition]
            position = rule.action_location.treeposition
            rule.action_location.original_treeposition = position
            rule.action_location.treeposition = ()
            root[position].rules += [rule]
        except IndexError:
            rule.action_location.original_treeposition = rule.action_location.treeposition
            root.rules += [rule]
        return root

class CompressedLTAGLoader(SpinalLTAGLoader):
    """Class to load SpinalLTAG's from a compressed json format"""

    def __init__(self, filename=None):
        return super(CompressedLTAGLoader, self).__init__(filename)

    def load(self, limit=None):
        with open(self.filename) as json_file:
            trees = json.loads(json_file.read())

        if limit is not None:
            spinal_ltags = [tree for tree_dict in trees[:limit] for tree in self.parse_ltags_from_dict(tree_dict)]
        else:
            spinal_ltags = [tree for tree_dict in trees for tree in self.parse_ltags_from_dict(tree_dict)]
        return spinal_ltags

    def lexicalize_tree(self, root, lexicalization_dict):
        """
        Takes an unlexicalized tree and a dictionary of {word: count} representing words that
        lexicalize this tree and the frequency with which they do so

        Returns a list of all possible lexicalized trees, storing the tree count and the lexicalization count for
        use in calculating a tree probability
        """

        trees = []
        tree_count = sum(lexicalization_dict.values())
        for word, count in lexicalization_dict.items():
            tree = root.copy(deep=True)

            # Add lexicalization as spine's child
            node = tree
            while len(node) > 0:
                node = node[-1]
            node.append(word)

            tree.lexicalization_count = count
            tree.tree_count = tree_count
            trees.append(tree)

        return trees

    def parse_ltags_from_dict(self, tree_dict):
        """Given a dictionary, returns a list of parsed Spinal LTAGs"""

        spine = re.sub('[()]', '', tree_dict['spine'])
        node_labels = spine.split()
        nodes = [SpinalLTAG(label, children=[], tree_type=tree_dict['tree_type']) for label in node_labels]

        for current, next in pairwise(nodes):
            current.append(next)

        root = nodes[0]
        root.predicate = tree_dict['predicate']
        root.roleset_id = tree_dict['roleset_id']
        root.num_args = tree_dict['num_args']
        root.tree_id = tree_dict['tree_id']

        # Create rules and assign them to nodes in tree
        for rule_dict in tree_dict['rules']:
            rule_dict['treeposition'] = ".".join(['0'] + [str(i) for i in rule_dict['treeposition']])
            rule = Rule.from_dict(rule_dict)

            attach_key = str((rule_dict['treeposition'], str(rule.action_location.slot)))
            if attach_key in tree_dict['attach_counts']:
                rule.attach_counts = tree_dict['attach_counts'][attach_key]

            root = self.add_rule_to_tree(root, rule)

        trees = self.lexicalize_tree(root, tree_dict['lexicalization'])
        return trees

class UncompressedSpinalLTAGLoader(SpinalLTAGLoader):
    def __init__(self, filename=None): 
        return super(UncompressedSpinalLTAGLoader, self).__init__(filename)

    def load(self, limit=None):
        with open(self.filename) as json_file:
            trees = json.loads(json_file.read())

        if limit is not None:
            spinal_ltags = [self.parse_ltag_from_dict(tree_dict) for tree_dict in trees[:limit]]
        else:
            spinal_ltags = [self.parse_ltag_from_dict(tree_dict) for tree_dict in trees]
        return spinal_ltags

    def parse_ltag_from_dict(self, tree_dict):
        """Given a dict, returns a parsed Spinal LTAG"""

        spine = re.sub('[()]', '', tree_dict['spine'])
        node_labels = spine.split()
        nodes = [SpinalLTAG(label, children=[], tree_type=tree_dict['type']) for label in node_labels]
        nodes.append(tree_dict['terminal'])

        for current, next in pairwise(nodes):
            current.append(next)

        root = nodes[0]
        root.predicate = tree_dict['predicate']
        root.roleset_id = tree_dict['roleset_id']
        root.num_args = tree_dict['num_args']
        root.tree_id = tree_dict['tree_id']
        root.parent_id = tree_dict['parent_id']
        root.parent_attach_id = tuple(tree_dict['parent_attach_id']) if tree_dict['parent_attach_id'] is not None else None

        # Create rules and assign them to nodes in tree
        for rule_dict in tree_dict['rules']:
            rule = Rule.from_dict(rule_dict)
            root = self.add_rule_to_tree(root, rule)

        return root
