from spinal.spinal_grammar import SpinalGrammar
from state import State

class SpinalState(State):
    """
    Spinal State implements the state interface with the LTAG-Spinal formalism as the tree + actions
    """
    def __init__(self, exploration_constant, tree, grammar, reward):
        self.explorationconstant = exploration_constant
        self.tree = tree
        self.grammar = grammar
        self.reward = reward

    def clone(self):
        """
        Returns a deep copy of the state
        """
        treeclone = None
        if self.tree is not None:
            treeclone = self.tree.copy(True)
        s = SpinalState(self.explorationconstant, treeclone, self.grammar, self.reward)
        return s

    def execute_action(self, action_idx):
        """
        Modifies this state's tree by applying the given action
        """
        if action_idx is None or (isinstance(action_idx, int) and self.actions()[action_idx] is None):
            pass
        else:                
            if isinstance(action_idx, TreeAction):
                self.tree = action_idx.execute(self.tree)
            else:
                self.tree = self.actions()[action_idx].execute(self.tree)
        return self

    def get_initial_actions(self):
        """
        Returns the set of actions that begin with the grammar's start symbol
        """
        initial_actions = []
        for tree in self.grammar.tree_dict[self.grammar.start]:
            initial_actions.append(InitialAction(tree))
        return initial_actions

    def get_possible_actions(self):
        return self.actions()
        
    def actions(self):
        """
        Returns the actions that can be applied in this state
        """

        if self.tree is None:
            return self.get_initial_actions()

        sub_actions = []

        for pos in self.tree.open_actions():
            for tree in self.grammar.tree_dict[pos]:
                sub_actions.append(SubstituteAction(tree))

        return sub_actions

    def get_value(self):
        """
        Evaluates this state according to this state's reward function
        """
        return self.reward.evaluate(self.tree)

    def sentence(self):
        """
        Returns the sentence stored by this state's tree
        """
        if self.tree is None:
            return ""
        return str(self.tree)

    def is_terminal(self):
        """
        Returns whether or not this state's tree is terminal
        """
        if self.tree is None:
            return False
        else:
            return self.tree.terminal_tree()

    def exploration_constant(self):
        return self.explorationconstant

    def __repr__(self):
        return "<SpinalState: %s>" % (str(self.tree))

class TreeAction(object):
    """
    Stores a tree that can be applied as an action to another tree to generate a new tree
    """
    def __init__(self, tree):
        self.tree = tree

    def execute(self, tree):
        raise NotImplementedError

class SubstituteAction(TreeAction):
    """
    Represents a tree action that generates a new tree via substitution
    """

    def execute(self, tree):
        return tree.attach(self.tree)[0]

    def __repr__(self):
        return "<SubstituteAction: %s>" % (str(self.tree))

class InitialAction(TreeAction):
    """
    Represents a tree action that generates an entirely new tree (the tree action becomes the tree)
    """

    def execute(self, tree):
        return self.tree

    def __repr__(self):
        return "<InitialAction: %s>" % (str(self.tree))

def demo():
    #pos_whitelist  = set(["S", "NP", "NN", "VP", "VB", "VBD", ".", ",", "NNP", "DT"])
    grammar = SpinalGrammar.from_file()
    print(grammar.tree_dict['S'])
    state = SpinalState(0.5, None, grammar, None)
    while not state.is_terminal():
        print(state)
        state.execute_action(0)
        state.tree.draw()
    print(state)
    state.tree.draw()

if __name__ == "__main__":
    demo()



