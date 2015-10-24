import re
from collections import defaultdict
from spinal.ltag_spinal import SpinalLTAG

class SpinalGrammar(object):
    """
    Stores the grammar formed by a full set of LTAG-Spinal elementary trees
    """
    def __init__(self, trees, start_symbol):
        self.trees = trees
        self.start = start_symbol
        self.tree_dict = defaultdict(list)
        for tree in trees:
            self.tree_dict[tree.label()].append(tree)

    def __repr__(self):
        return "<SpinalGrammar: start symbol=%s, num trees=%d>" % (self.start, len(self.trees))

    @classmethod
    def from_file(cls, filename="trees.dat", tree_begin="TREE BEGIN", tree_end="TREE END", pos_whitelist=None, tree_whitelist=None, limit=5000):
        """
        Loads the grammar from a file and returns a SpinalGrammar object
        During loading, filters according to a pos whitelist and a tree whitelist
        """

        if pos_whitelist is None:
            pos_whitelist = set(["S", "NP", "NN", "VP", "VB", "VBD", ".", "DT"])

        if tree_whitelist is None:
            tree_whitelist = set(["NP \(NN", "^S \(VP", "VP \(VB", "VP \(VBD", "\.", "^\(DT"])

        with open(filename) as f:
            trees = []
            tree = []

            for raw_line in f:
                line = raw_line.strip()

                if len(trees) > limit:
                    break

                if line == tree_begin:
                    tree = []
                elif line == tree_end:
                    trees.append(tree)
                else:
                    tree.append(line)

        spinal_ltags = []
        tree_set = set()
        for tree in trees:
            spinal_ltag = SpinalLTAG.from_tree_string_array(tree)

            if len(spinal_ltag.pos_set() - pos_whitelist) > 0:
                continue
            if str(spinal_ltag) in tree_set:
                continue

            cont = True
            for tree_allowed in tree_whitelist:
                if re.search(tree_allowed, str(spinal_ltag)) is not None:
                    cont = False
            if cont:
                continue

            spinal_ltags.append(spinal_ltag)
            tree_set.add(str(spinal_ltag))

        return SpinalGrammar(spinal_ltags, "S")
