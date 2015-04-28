from edu.upenn.cis.propbank_shen import *
from edu.upenn.cis.spinal import *
from collections import deque
import re

propbank = Propbank()

def get_propbank_label(attach):
    label = ""
    if attach.getParent().getPASLoc() in propbank:
        annotation = propbank[attach.getParent().getPASLoc()]
        pastruct = annotation.getPAStruct()
        matching_args = [a for a in pastruct.getArgs() if a.getLocation().toString() == attach.getChild().getSpan().toString()]

        if len(matching_args) > 0:
            label = matching_args[0].arg_label.toString()
            if label == "ARGM" and matching_args[0].mod_label is not None:
                label += "-" + matching_args[0].mod_label.toString()
    return label


def print_tree(tree_str):
    sentence = Sentence(tree_str)

    try:
        queue = [sentence.getRoot()]
        while len(queue) > 0:
            elem = queue.pop()
            attachments = elem.getAttachments()

            print "TREE BEGIN"
            print "type: %s spine: %s terminal: %s" % (elem.getTypeAsString(), str(elem.getSpine()), elem.getTerminal())
            if elem.isRoot() and elem.getPASLoc() in propbank:
                annotation = propbank[elem.getPASLoc()]
                pastruct = annotation.getPAStruct()
                print "predicate: " + pastruct.getLemma()

            for attach in attachments:
                rule_string = attach.getGeneralString()
                semantic_role = get_propbank_label(attach)
                print rule_string + ", semantic_role: " + semantic_role

                queue.append(attach.getChild())
            print "TREE END"
    except SkippedSentenceException:
        return

class SpinalTreeBankLoader(object):
    def __init__(self):
        self.directory = "/Users/piffle/Desktop/spinalapi/spinalapi/ltagtb/"
        self.filenames = [
            "derivation.sec0-1.v01",
            "derivation.devel.v01",
            "derivation.test.v01",
            "derivation.train.v01"
        ]
        self.tree_begin = "^\d+ \d+ \d+$"
        self.tree_end = "^\s$"

    def process(self, tree_function):
        for filename in self.filenames:
            with open(self.directory + filename) as f:
                tree = ""
                for line in f:

                    m_begin = re.search(self.tree_begin, line)
                    m_end = re.search(self.tree_end, line)

                    if m_begin:
                        tree = line
                    elif m_end:
                        tree_function(tree)
                    else:
                        tree += line

if __name__ == "__main__":
    SpinalTreeBankLoader().process(print_tree)    
