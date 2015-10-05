import nltk
from nltk.tree import Tree, ParentedTree
from collections import deque, defaultdict
from itertools import tee
import json
import re
import random

class SpinalLTAGLoader(object):
    """
    Class to load a treebank in the generalized tree format that was converted from Libin Shen's treebank
    """

    def __init__(self, filename="trees.json", pos_whitelist=None):
        self.filename = filename

        if pos_whitelist is None:
            pos_whitelist = set()
        self.pos_whitelist = pos_whitelist

    def load_uncompressed(self, limit=None):
        with open(self.filename) as json_file:
            trees = json.loads(json_file.read())

        if limit is not None:
            spinal_ltags = [SpinalLTAG.from_uncompressed_dict(tree_dict) for tree_dict in trees[:limit]]
        else:
            spinal_ltags = [SpinalLTAG.from_uncompressed_dict(tree_dict) for tree_dict in trees]
        return spinal_ltags

    def load(self, limit=None):
        """
        Returns a list of SpinalLTAGs
        """
        with open('output/compressed_trees.json') as json_file:
            trees = json.loads(json_file.read())

        if limit is not None:
            spinal_ltags = [tree for tree_dict in trees[:limit] for tree in SpinalLTAG.from_compressed_dict(tree_dict)]
        else:
            spinal_ltags = [tree for tree_dict in trees for tree in SpinalLTAG.from_compressed_dict(tree_dict)]
        return spinal_ltags
        

class SpinalLTAG(ParentedTree):
    """
    Represents a Spinal LTAG as described in Libin Shen's thesis. 
    See: http://www.cis.upenn.edu/~xtag/spinal/
    Written as a subclass of nltk.Tree for Natural Language purposes
    """

    def __init__(self, name, **kwargs):

        self.rules = kwargs.get('rules', [])
        self.children = kwargs.get('children', [])
        self.tree_type = kwargs.get('tree_type')
        self.predicate = kwargs.get('predicate')
        self.tree_id = kwargs.get('tree_id')
        self.parent_id = kwargs.get('parent_id')
        self.parent_attach_id = kwargs.get('parent_attach_id')
        self.attached = kwargs.get('attached', False)
        self.roleset_id = kwargs.get('roleset_id')
        self.num_args = int(kwargs.get('num_args')) if kwargs.get('num_args' ) is not None else None
        self.semantic_role = kwargs.get('semantic_role')

        if "^" in name:
            name = name[:-1]
            self.foot = True
        else:
            self.foot = False

        super(SpinalLTAG, self).__init__(name, children=self.children)

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
        pos.add(self.label())
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
            rules = current.pos_rule_dict()[att_tree.label()]
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
                    print(loc)
                    assert False

                # Perform attachment
                insertion_node.insert(attachment_location, att_tree)

                # Remove the attachment rule just used
                root[current.treeposition()].rules = [r for r in current.rules if r != rule]

                trees.append(root)

        return trees

    def amr_semantics(self):
        nodes = set()
        edges = set()

        queue = deque([self])
        current_predicate = self.predicate
        while len(queue) > 0:
            current = queue.popleft()

            if current.predicate is not None:
                current_predicate = current.predicate
                nodes.add(current_predicate)

            if current.semantic_role is not None:
                edges.add((current.semantic_role, current.treeposition(), current_predicate))

            for r in current.rules:
                if r.semantic_role is not None:
                    edges.add((r.semantic_role, None, current_predicate))

            for c in current:
                if isinstance(c, SpinalLTAG):
                    queue.append(c)

        return nodes, edges

    def fol_semantics(self):
        nodes, edges = self.amr_semantics()
        arg_dict = {}

        for pred in nodes:
            arg_dict[pred] = []

        for (role, arg, pred) in edges:
            if role[-1].isdigit():
                if pred in arg_dict:
                    arg_dict[pred].append((role, arg))
                else:
                    arg_dict[pred] = []
                    arg_dict[pred].append((role, arg))

        semantics = set()
        entities = set()
        for pred, arg_list in arg_dict.items():
            sorted_arg_list = sorted(arg_list, key=lambda x: x[0])
            arg_string = ", ".join([self.entity_from_treeposition(arg, i=i) for i, (role, arg) in enumerate(sorted_arg_list)])
            pred_string = "%s(%s)" % (pred, arg_string)
            semantics.add(pred_string)

            for i, (role, arg) in enumerate(sorted_arg_list):
                entities.add(self.entity_from_treeposition(arg, i=i))
                if arg is not None:
                    semantics.add(self.predicate_from_treeposition(arg, i=i))

        return entities, semantics

    def entity_from_treeposition(self, treepos, i=0):
        if treepos is None:
            return "ARG%i" % i
        else:
            return remove_chars(self[treepos].label() + "_" + str(treepos), "() ").replace(",", "_")

    def predicate_from_treeposition(self, treepos, i=0):
        grammar = "SimpleNoun: {(<NNP>|<NN>|<NNS>)*}"
        cp = nltk.RegexpParser(grammar)
        chunked = cp.parse(self[treepos])
        simple_nouns = [c for c in chunked.subtrees(lambda tree: tree.label() == "SimpleNoun")]

        if len(simple_nouns) > 0:
            pred_name = "_".join(simple_nouns[0].leaves())
            pred_string = "%s(%s)" % (pred_name, self.entity_from_treeposition(treepos, i=i))
        else:
            pred_string = None

        return pred_string

    @classmethod
    def convert(cls, val):
        if isinstance(val, Tree):
            children = [cls.convert(child) for child in val]
            if isinstance(val, SpinalLTAG):
                return cls(
                    val.label(),
                    children=children, 
                    tree_type=val.tree_type,
                    predicate=val.predicate,
                    rules=val.rules,
                    attached=val.attached,
                    semantic_role=val.semantic_role,
                    roleset_id=val.roleset_id,
                    num_args=val.num_args,
                    tree_id=val.tree_id,
                    parent_id=val.parent_id,
                    parent_attach_id=val.parent_attach_id,
                )
            else:
                return cls(val.label(), children=children)
        else:
            return val

    @classmethod
    def from_uncompressed_dict(cls, tree_dict):
        # Create SpinalLTAG from spine
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

            try:
                root[rule.action_location.treeposition]
                position = rule.action_location.treeposition
                rule.action_location.treeposition = ()
                root[position].rules.append(rule)
            except IndexError:
                root.rules.append(rule)

        return root

    @classmethod
    def from_compressed_dict(cls, tree_dict):
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
            #print(attach_key, tree_dict['attach_counts'])
            if attach_key in tree_dict['attach_counts']:
                rule.attach_counts = tree_dict['attach_counts'][attach_key]

            try:
                root[rule.action_location.treeposition]
                position = rule.action_location.treeposition
                rule.action_location.treeposition = ()
                root[position].rules.append(rule)
            except IndexError:
                root.rules.append(rule)

        trees = []
        tree_count = sum(tree_dict['lexicalization'].values())
        for word, count in tree_dict['lexicalization'].items():
            tree = root.copy(deep=True)
            if len(tree) == 0:
                tree.append(word)
            else:
                tree[-1].append(word)
            tree.lexicalization_count = count
            tree.tree_count = tree_count
            trees.append(tree)
        return trees

class Rule(object):
    def __init__(self, rule_type, pos, action_location, action_id=None, semantic_role=None, role_desc=None):
        self.rule_type = rule_type
        self.pos = pos
        self.action_location = action_location
        self.action_id = action_id
        self.semantic_role = semantic_role
        self.role_desc = role_desc

    def __repr__(self):
        return "<Rule: %s %s on %s, slot %s, order %s, semantic_role %s>" % (self.rule_type, self.pos, self.action_location.treeposition, self.action_location.slot, self.action_location.order, self.semantic_role)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_dict(self):
        return {
            'treeposition': self.action_location.treeposition,
            'slot': self.action_location.slot,
            'order': self.action_location.order,
            'rule_type': self.rule_type,
            'pos': self.pos,
            'semantic_role': self.semantic_role,
            'role_desc': self.role_desc
        }

    @classmethod
    def from_dict(cls, rule_dict):
        treeaddress = TreeAddress.from_string(rule_dict['treeposition'])
        action_location = ActionLocation(treeaddress, int(rule_dict['slot']), int(rule_dict['order']))
        return Rule(rule_dict['rule_type'], rule_dict['pos'], action_location, action_id=rule_dict.get('attach_id'), semantic_role=rule_dict.get('semantic_role'), role_desc=rule_dict.get('desc'))

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
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def remove_chars(s, chars):
    return s.translate(str.maketrans("", "", chars))

def demo():
    pos_whitelist  = set(["S", "NP", "NN", "VP", "VB", "VBD", ".", ",", "NNP", "DT"])
    tree_loader = SpinalLTAGLoader(pos_whitelist=pos_whitelist)
    trees = tree_loader.load_uncompressed()
    print(len(trees))

    tree_dict = defaultdict(list)
    for tree in trees:
        tree_dict[tree.label()].append(tree)

    root = tree_dict["S"][0]
    print(root)

    while not root.terminal_tree():
        print(root)
        actions = root.open_actions()
        pos = actions[0]
        att_tree = tree_dict[pos][0]
        root = root.attach(att_tree)[0]
        root.draw()

    nodes, edges = root.amr_semantics()
    print(nodes, edges)
    print(root.fol_semantics())


def selectFromWeightedList(weightedlist, heuristic=lambda x: 1):
    if all(not isinstance(x, tuple) for x in weightedlist):
        weightedlist = [(x,heuristic(x)) for x in weightedlist]
    weights = [p for a,p in weightedlist]
    choice = random.random() * sum(weights)
    for i, p in weightedlist:
        choice -= p
        if choice <= 0:
            return i
    assert False

def compressed_demo():
    tree_loader = SpinalLTAGLoader()
    trees = tree_loader.load()

    tree_dict = defaultdict(list)
    for tree in trees:
        tree_dict[tree.label()].append(tree)

    root = selectFromWeightedList([(t, t.lexicalization_count) for t in tree_dict["S"]])
    print(root)

    while not root.terminal_tree():
        print(root)
        actions = root.open_actions()
        pos = random.choice(actions)
        print(pos)
        print(actions)
        att_tree = selectFromWeightedList([(att_tree, att_tree.lexicalization_count) for att_tree in tree_dict[pos]])
        print('attaching', att_tree, att_tree.tree_count, att_tree.lexicalization_count)
        root = random.choice(root.attach(att_tree))
        print(root)
        print(root.fol_semantics())
        root.draw()

    return trees

if __name__ == "__main__":
    compressed_demo()
