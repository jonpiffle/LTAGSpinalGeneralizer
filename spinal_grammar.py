import re, os, pickle
from collections import defaultdict
from spinal.ltag_spinal import SpinalLTAG
from spinal.spinal_loader import CompressedLTAGLoader

class SpinalGrammar(object):
    """
    Stores the grammar formed by a full set of LTAG-Spinal elementary trees
    """
    def __init__(self, trees, start_symbol, limit=None):
        self.trees = trees
        self.start = start_symbol
        self.tree_dict = defaultdict(list)
        for tree in trees:
            if limit is None or len(self.tree_dict[tree.label()]) < limit:
                self.tree_dict[tree.label()].append(tree)

    def __repr__(self):
        return "<SpinalGrammar: start symbol=%s, num trees=%d>" % (self.start, len(self.trees))

    @classmethod
    def from_file(cls, tree_loader_cls=CompressedLTAGLoader, filename="output/compressed_trees.json", pos_whitelist=None, tree_whitelist=None, limit=None, update=False):
        """
        Loads the grammar from a file and returns a SpinalGrammar object
        During loading, filters according to a pos whitelist and a tree whitelist
        """
        pickle_filename = filename.split(".")[0] + ".pickle"
        if not os.path.exists(pickle_filename) or update:

            if pos_whitelist is None:
                pos_whitelist = set(["S", "NP", "NN", "VP", "VB", "VBD", "DT", 'JJ', 'ADJP', 'NNS', 'IN', 'JJR', 'JJS', 'NNP', 'PRN'])

            if tree_whitelist is None:
                tree_whitelist = set(["^\(NN [a-z]+", "^\(NP \(NN", "^\(S \(VP", "^\(DT", "^\(ADJP", "^\(JJ"])

            tree_loader = tree_loader_cls(filename)

            final_trees = [] 
            trees = tree_loader.load()
            for tree in trees:
                if len(tree.pos_set() - pos_whitelist) > 0:
                    continue

                if all(re.search(t_allowed, str(tree)) is None for t_allowed in tree_whitelist):
                    continue

                final_trees.append(tree)
            pickle.dump(final_trees, open(pickle_filename, 'wb'))
        else:
            final_trees = pickle.load(open(pickle_filename, 'rb'))

        return SpinalGrammar(final_trees, "S", limit=limit)
