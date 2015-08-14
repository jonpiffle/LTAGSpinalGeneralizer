from nltk.tree import Tree, ParentedTree
from collections import deque, defaultdict
from itertools import izip, tee
import re

class SpinalLTAGLoader(object):
    """
    Class to load a treebank in the generalized tree format that I converted Libin Shen's treebank to
    """

    def __init__(self, filename="trees.dat", tree_begin="TREE BEGIN", tree_end="TREE END", pos_whitelist=None):
        self.filename = filename
        self.tree_begin = tree_begin
        self.tree_end = tree_end

        if pos_whitelist is None:
            pos_whitelist = set()
        self.pos_whitelist = pos_whitelist

    def load(self):
        """
        Returns a list of SpinalLTAGs
        """

        with open(self.filename) as f:
            trees = []
            tree = []

            for raw_line in f:
                line = raw_line.strip()

                if line == self.tree_begin:
                    tree = []
                elif line == self.tree_end:
                    trees.append(tree)
                else:
                    tree.append(line)

        spinal_ltags = []
        for tree in trees:
            spinal_ltag = SpinalLTAG.from_tree_string_array(tree)

            if len(spinal_ltag.pos_set() - pos_whitelist) == 0:
                spinal_ltags.append(spinal_ltag)
        return spinal_ltags

class SpinalLTAG(ParentedTree):
    """
    Represents a Spinal LTAG as described in Libin Shen's thesis. 
    See: http://www.cis.upenn.edu/~xtag/spinal/
    Written as a subclass of nltk.Tree for Natural Language purposes
    """

    def __init__(self, name, children=None, tree_type=None, predicate=None, rules=None, attached=False, semantic_role=None):
        if rules is None:
            rules = []

        self.rules = rules
        self.tree_type = tree_type
        self.predicate = predicate
        self.attached = attached
        self.semantic_role = semantic_role

        if "^" in name:
            name = name[:-1]
            self.foot = True
        else:
            self.foot = False

        super(SpinalLTAG, self).__init__(name, children=children)

    def terminal_tree(self):
        """
        No applicable rules left in any part of this tree
        """

        return len(self.all_applicable_rules()) == 0

    def all_applicable_rules(self):
        """
        Gets all applicable rules in this tree through a depth-first search
        """

        rules = self.applicable_rules()
        for child in self:
            if isinstance(child, SpinalLTAG):
                rules += child.all_applicable_rules()
        return rules

    def applicable_rules(self):
        """
        Gets all rules that can be applied to this node in its current state.
        Currently forcing nodes that insert in the same slot and location to be attached in ascending order
        """

        app_rules = []

        action_dict = defaultdict(list)
        for r in self.rules:
            action_dict[tuple([r.action_location.treeposition, r.action_location.slot])].append(r)

        for k, rule_list in action_dict.items():
            r = sorted(rule_list, key=lambda r: r.action_location.order)[0]
            app_rules.append(r)

        return app_rules

    def open_actions(self):
        """
        Returns a list of POS's that can be attached at any point in this tree
        """

        return list(set([r.pos for r in self.all_applicable_rules()]))

    def spine_index(self):
        """
        Returns the index of the immediate child that is along the same spine as this node
        """

        count = 0
        for child in self:
            if type(child) == str or not child.attached:
                return count
            else:
                count += 1
        assert False

    def pos_rule_dict(self):
        """
        Returns a dictionary mapping:
            POS: [rule]
        where that part of speech can be attached
        """

        rule_dict = defaultdict(list)
        for r in self.applicable_rules():
            rule_dict[r.pos].append(r)
        return rule_dict

    def pos_set(self):
        pos = set()
        pos.add(self.node)
        for r in self.rules:
            pos.add(r.pos)

        for child in self:
            if isinstance(child, SpinalLTAG):
                pos.update(child.pos_set())

        return pos

    def attach(self, att_tree):
        """
        Does a breadth first search through the tree
        For each location at each node that it is possible to attach att_tree: 
            1. The entire tree will be copied
            2. att_tree will be attached at the possible location
            3. The resulting tree will be appended to trees
        """

        trees = []
        queue = deque([self])

        # Breadth first search
        while len(queue) > 0:
            current = queue.popleft()

            # Add all non-leaf children to queue to continue BFS
            for child in current:
                if isinstance(child, SpinalLTAG):
                    queue.append(child)

            # Get all attachment locations for the POS specified by the root of att_tree
            rules = current.pos_rule_dict()[att_tree.node]
            if len(rules) == 0:
                continue

            # Make a new tree for each possible location
            for rule in rules:
                loc = rule.action_location
                root = current.root().copy(True)
                current = root[current.treeposition()]
                att_tree = att_tree.copy(True)
                att_tree.semantic_role = rule.semantic_role
                att_tree.attached = True

                # Find the node that is specified by the attachment rule
                insertion_node = current[loc.treeposition]
                if insertion_node is None:
                    continue

                # Count siblings on either side of the spine to determine attachment position
                left_siblings = insertion_node[:current.spine_index()]
                right_siblings = insertion_node[current.spine_index() + 1:]

                # Attach to the left of the spine
                if loc.slot == 0:
                    if len(left_siblings) == loc.order:
                        attachment_location = loc.order
                    else:
                        continue

                # Attach to the right of the spine
                elif loc.slot == 1:
                    if len(right_siblings) == loc.order:
                        attachment_location = loc.order + 1 + current.spine_index()
                    else:
                        continue

                # Only slot options are 0 (left) and 1 (right)
                else:
                    assert False

                # Perform attachment
                insertion_node.insert(attachment_location, att_tree)

                # Remove the attachment rule just used
                root[current.treeposition()].rules = [r for r in current.rules if r != rule]

                trees.append(root)

        return trees

    def amr_name(self):
        if self.predicate is not None:
            return self.predicate
        else:
            return " ".join(self.leaves())

    def amr_semantics(self):
        nodes = set([])
        edges = set([])

        queue = deque([self])
        current_predicate = self.predicate
        while len(queue) > 0:
            current = queue.popleft()

            if current.predicate is not None:
                current_predicate = current.predicate
                nodes.add(current_predicate)

            if current.semantic_role is not None:
                edges.add((current.semantic_role, current.treeposition(), current_predicate))

            for c in current:
                if isinstance(c, SpinalLTAG):
                    queue.append(c)

        return nodes,edges

    def fol_semantics(self):
        nodes, edges = self.amr_semantics()
        arg_dict = {}
        for pred in nodes:
            arg_dict[pred] = []

        for (role, arg, pred) in edges:
            if role[-1].isdigit():
                arg_dict[pred].append((role, arg))

        semantics = set([])
        for pred, arg_list in arg_dict.items():
            arg_string = ", ".join([self.entity_from_treeposition(arg) for role, arg in sorted(arg_list, key=lambda x: x[0])])
            pred_string = "%s(%s)" % (pred, arg_string)
            semantics.add(pred_string)

            for role, arg in arg_list:
                semantics.add(self.predicate_from_treeposition(arg))

        return semantics

    def entity_from_treeposition(self, treepos):
        return self[treepos].node + "_" + str(treepos).translate(None, "() ").replace(",", "_")

    def predicate_from_treeposition(self, treepos):
        pred_name  = "_".join(self[treepos].leaves())
        pred_string = "%s(%s)" % (pred_name, self.entity_from_treeposition(treepos))
        return pred_string

    @classmethod
    def convert(cls, val):
        if isinstance(val, Tree):
            children = [cls.convert(child) for child in val]
            if isinstance(val, SpinalLTAG):
                return cls(val.node, children, tree_type=val.tree_type, predicate=val.predicate, rules=val.rules, attached=val.attached, semantic_role=val.semantic_role)
            else:
                return cls(val.node, children)
        else:
            return val

    @classmethod
    def from_tree_string_array(cls, tree_string_array):
        # Parse basic tree information (type, spine, and terminal)
        tree_def = tree_string_array[0]
        m = re.search('type:(.*) spine:(.*) terminal:(.*)', tree_def)

        tree_type = m.group(1)[1:]
        spine = m.group(2)[1:]
        terminal = m.group(3)[1:]

        # Parse predicate associated with tree, if exists
        if len(tree_string_array) > 1 and tree_string_array[1] is not None and "predicate:" in tree_string_array[1]:
            predicate_match = re.search("predicate: (.*)", tree_string_array[1])
            predicate = predicate_match.group(1)
            rule_strs = tree_string_array[2:]
        else:
            predicate = None
            rule_strs = tree_string_array[1:]

        # Create SpinalLTAG from spine
        spine = re.sub('[()]', '', spine)
        node_labels = spine.split()

        nodes = [SpinalLTAG(label, children=[], tree_type=tree_type) for label in node_labels]
        nodes.append(terminal)

        for current, next in pairwise(nodes):
            current.append(next)

        root = nodes[0]
        root.predicate = predicate

        # Create rules and assign them to nodes in tree
        for rule_str in rule_strs:
            rule = Rule.from_string(rule_str)

            try:
                root[rule.action_location.treeposition]
                position = rule.action_location.treeposition
                rule.action_location.treeposition = ()
                root[position].rules.append(rule)
            except IndexError:
                root.rules.append(rule)

        return root

class Rule(object):
    def __init__(self, rule_type, pos, action_location, semantic_role=None):
        self.rule_type = rule_type
        self.pos = pos
        self.action_location = action_location
        self.semantic_role = semantic_role

    def __repr__(self):
        return "<Rule: %s %s on %s, slot %s, order %s, semantic_role %s>" % (self.rule_type, self.pos, self.action_location.treeposition, self.action_location.slot, self.action_location.order, self.semantic_role)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def from_string(cls, string):
        att_adj = re.search("(att|adj) (.*), on (.*), slot (.*), order (.*), semantic_role:(.*)", string)
        att_crd = re.search("(att|crd) (.*), on (.*), semantic_role:(.*)", string)

        if att_adj:
            rule_type = att_adj.group(1)
            pos = att_adj.group(2)
            treeposition = TreeAddress.from_string(att_adj.group(3))
            slot = int(att_adj.group(4))
            order = int(att_adj.group(5))
            semantic_role = att_adj.group(6).strip()
            if semantic_role == "":
                semantic_role = None
        elif att_crd:
            rule_type = att_crd.group(1)
            pos = att_crd.group(2)
            treeposition = TreeAddress.from_string(att_crd.group(3))
            slot = 0
            order = 0
            semantic_role = att_crd.group(4).strip()
            if semantic_role == "":
                semantic_role = None
        else:
            assert False, string

        action_location = ActionLocation(treeposition, slot, order)
        return Rule(rule_type, pos, action_location, semantic_role)

class TreeAddress(tuple):
    def __new__(cls, lst):
        return super(TreeAddress, cls).__new__(cls, tuple(lst))

    @classmethod
    def from_string(cls, string):
        lst = [int(i) for i in string.split(".")[1:]]
        return TreeAddress(lst)

class ActionLocation(object):
    def __init__(self, treeposition, slot, order):
        self.treeposition = treeposition
        self.slot = slot
        self.order = order

    def __repr__(self):
        return "<Loc: %s %d %d>" % (str(self.treeposition), self.slot, self.order)

    def __eq__(self, other):
        return (isinstance(other, self.__class__)
            and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

if __name__ == "__main__":
    pos_whitelist  = set(["S", "NP", "NN", "VP", "VB", "VBD", ".", ",", "NNP", "DT"])
    tree_loader = SpinalLTAGLoader(filename="trees.dat")
    trees = tree_loader.load()
    print len(trees)

    tree_dict = defaultdict(list)
    for tree in trees:
        tree_dict[tree.node].append(tree)

    root = tree_dict["S"][0]
    print root

    while not root.terminal_tree():
        print root
        actions = root.open_actions()
        pos = actions[0]
        att_tree = tree_dict[pos][0]
        root = root.attach(att_tree)[0]

    nodes, edges = root.amr_semantics()
    print nodes, edges
    print root.fol_semantics()
    root.draw()
