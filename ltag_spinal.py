import nltk, random
from collections import deque, defaultdict
from nltk.tree import Tree, ParentedTree
from spinal.spinal_loader import *

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
        """

        '''
        app_rules = []
        action_dict = defaultdict(list)
        for r in self.rules:
            action_dict[tuple([r.action_location.treeposition, r.action_location.slot])].append(r)

        for k, rule_list in action_dict.items():
            r = rule_list[0]
            app_rules.append(r)

        return app_rules 
        '''
        return self.rules

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
                    attachment_location = loc.slot

                # Attach to the right of the spine
                elif loc.slot == 1:
                    attachment_location = loc.slot + current.spine_index()

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

class Rule(object):
    def __init__(self, rule_type, pos, action_location, action_id=None, semantic_role=None, role_desc=None):
        self.rule_type = rule_type
        self.pos = pos
        self.action_location = action_location
        self.action_id = action_id
        self.semantic_role = semantic_role
        self.role_desc = role_desc

    def __repr__(self):
        return "<Rule: %s %s on %s, slot %s, semantic_role %s>" % (self.rule_type, self.pos, self.action_location.treeposition, self.action_location.slot, self.semantic_role)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(str(pos) + str(action_location) + str(semantic_role))

    def to_dict(self):
        return {
            'treeposition': self.action_location.original_treeposition,
            'slot': self.action_location.slot,
            'rule_type': self.rule_type,
            'pos': self.pos,
            'semantic_role': self.semantic_role,
            'role_desc': self.role_desc
        }

    @classmethod
    def from_dict(cls, rule_dict):
        treeaddress = TreeAddress.from_string(rule_dict['treeposition'])
        action_location = ActionLocation(treeaddress, int(rule_dict['slot']))
        return Rule(rule_dict['rule_type'], rule_dict['pos'], action_location, action_id=rule_dict.get('attach_id'), semantic_role=rule_dict.get('semantic_role'), role_desc=rule_dict.get('desc'))

class TreeAddress(tuple):
    def __new__(cls, lst):
        return super(TreeAddress, cls).__new__(cls, tuple(lst))

    @classmethod
    def from_string(cls, string):
        lst = [int(i) for i in string.split(".")[1:]]
        return TreeAddress(lst)

class ActionLocation(object):
    def __init__(self, treeposition, slot, original_treeposition=None):
        self.treeposition = treeposition
        self.slot = slot
        self.original_treeposition = original_treeposition

    def __repr__(self):
        return "<Loc: %s %d>" % (str(self.treeposition), self.slot)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self.__eq__(other)

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
    """Problem with ignoring order is having two rules in the same position for a tree"""
    tree_loader = CompressedLTAGLoader(filename='output/compressed_trees.json')
    trees = tree_loader.load(limit=1000)

    tree_dict = defaultdict(list)
    for tree in trees:
        tree_dict[tree.label()].append(tree)

    root = selectFromWeightedList([(t, t.lexicalization_count) for t in tree_dict["S"]])
    print(root)

    while not root.terminal_tree():
        print(root, root.num_args)
        print(root.all_applicable_rules())
        actions = root.open_actions()
        pos = random.choice(actions)
        print(pos)
        print(actions)
        att_tree = selectFromWeightedList([(att_tree, att_tree.lexicalization_count) for att_tree in tree_dict[pos]])
        print('attaching', att_tree, att_tree.tree_count, att_tree.lexicalization_count)
        root = random.choice(root.attach(att_tree))
        print(root)
        print(root.all_applicable_rules())
        print(root.fol_semantics())
        root.draw()

    return trees

if __name__ == "__main__":
    compressed_demo()
