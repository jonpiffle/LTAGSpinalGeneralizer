import json, os, re, sys, gc, glob
from collections import deque
sys.path.append('/Users/piffle/Documents/luna_workspace/spinal/bin')
from edu.upenn.cis.propbank_shen import *
from edu.upenn.cis.spinal import *
from itertools import product
from collections import defaultdict

propbank = Propbank()
tid = 0

def get_propbank_label(attach):
    label = None
    if attach.getParent().getPASLoc() in propbank:
        annotation = propbank[attach.getParent().getPASLoc()]
        pastruct = annotation.getPAStruct()
        matching_args = [a for a in pastruct.getArgs() if attach.getChild().getSpan().toString() in [s.toString() for s in a.getLocation().getAllWordSpans()]]
        if len(matching_args) > 0:
            label = matching_args[0].arg_label.toString()
            if label == "ARGM" and matching_args[0].mod_label is not None:
                label += "-" + matching_args[0].mod_label.toString()

    if label is not None:
        label = str(label)
    return label

def is_num_arg(arg):
    return arg is not None and arg[-1].isdigit()

def pos_map(string):
    for s in ['VBD', 'VBN', 'VBP', 'VBZ']:
        if s in string:
            string = string.replace(s, 'VB')
    return string

def parse_attachment(string):
    """Parses Libin Shen's 'Rule String' into an attachment dictionary"""
    att_adj = re.search("(att|adj) (.*), on (.*), slot (.*), order (.*)", string)
    att_adj_crd = re.search("(att|adj|crd) (.*), on (.*)", string)

    if att_adj:
        return {
            'rule_type': att_adj.group(1),
            'pos':  pos_map(att_adj.group(2)),
            'treeposition': att_adj.group(3),
            'slot': att_adj.group(4),
            'order': att_adj.group(5)
        }
    elif att_adj_crd:
        return {
            'rule_type': att_adj_crd.group(1),
            'pos': pos_map(att_adj_crd.group(2)),
            'treeposition': att_adj_crd.group(3),
            'slot': '0',
            'order': '0'
        }
    else:
        assert False, string

def print_tree(tree_str, output_file):
    global tid
    sentence = Sentence(tree_str)

    try:
        queue = [(sentence.getRoot(), None, None)]
        while len(queue) > 0:
            elem, parent_id, parent_attach_id = queue.pop()
            attachments = elem.getAttachments()

            tree = {
                'type': str(elem.getTypeAsString()),
                'spine': pos_map(str(elem.getSpine())),
                'terminal': str(elem.getTerminal()),
                'predicate': None,
                'roleset_id': None,
                'num_args': None,
                'propbank_loc': None,
                'propbank_annotation': None,
                'tree_id': tid,
                'parent_id': parent_id,
                'parent_attach_id': parent_attach_id,
            }

            # Lookup propbank entry for tree
            if elem.getPASLoc() in propbank:
                annotation = propbank[elem.getPASLoc()]

                # Creates a dict of {ARG0 -> Actor, ARG1 -> Agent, etc.}
                # A handful of annotation.getRoleSet()'s are null, so we just ignore
                arglist = {}
                if annotation.getRoleSet() is not None:
                    arglist = {str(r.getArgLabel()): str(r.getDescription()) for r in annotation.getRoleSet().getRoles()}
                else:
                    continue

                # Store propbank info about tree
                tree['propbank_loc'] = str(elem.getPASLoc())
                tree['propbank_annotation'] = str(propbank[elem.getPASLoc()])
                tree['predicate'] = annotation.getLemma()
                tree['roleset_id'] = annotation.getRoleSetId()
                tree['num_args'] = len(arglist)

            attachment_dicts = []
            for attach in attachments:
                # rule string is string storing info about attachment
                rule_string = attach.getGeneralString()
                attachment_dict = parse_attachment(rule_string)

                # parse semantic role of attachment from propbank (ARG0, ARGM, etc) 
                semantic_role = get_propbank_label(attach)
                attachment_dict['semantic_role'] = semantic_role

                # numbered arguments should have descriptions
                # if they don't something went wrong in parsing, so skip for now
                attachment_dict['desc'] = None
                if is_num_arg(semantic_role) and semantic_role not in arglist:
                    return None
                elif is_num_arg(semantic_role):
                    attachment_dict['desc'] = arglist[semantic_role]

                # only want to store mandatory verb attachments (semantic arguments) to reduce the 
                # number of unique trees for each verb
                if tree['predicate'] is None or is_num_arg(semantic_role):
                    attachment_dicts.append(attachment_dict)

                attach_id = (attachment_dict['treeposition'], attachment_dict['slot'])
                attachment_dict['attach_id'] = attach_id
                queue.append((attach.getChild(), tid, attach_id))
                
            tree['rules'] = attachment_dicts

            if tree['predicate'] is None or len(tree['rules']) > 0 and 'VBG' not in tree['spine']:

                # group rules by treeposition/slot
                new_rules = []
                attachment_groups = defaultdict(list)
                for attachment_dict in tree['rules']:
                    attachment_groups[attachment_dict['attach_id']].append(attachment_dict)

                # Reset order counts after filtering out non-semantic attachments
                for group in attachment_groups.values():
                    for i, rule in enumerate(sorted(group, key=lambda x: int(x['order']))):
                        rule['order'] = i
                        new_rules.append(rule)
                tree['rules'] = new_rules

                json_str = json.dumps(tree) + ","
                output_file.write(json_str + "\n")

                # update global tree counter
                tid += 1

    except SkippedSentenceException:
        return None

def process(section, output_filename='uncompressed_trees_u_rules.json', tree_function=print_tree):
    directory = "trees"

    for filename in glob.glob(directory + '/' + section +'_*.txt'):
        with open(filename, 'r') as f:
            tree = ""
            for line in f:
                tree += line
        with open(output_filename, 'a') as output_file:
            tree_function(tree, output_file=output_file)

def generate_semantics(tree_str, output_file="propbank_args.txt"):
    sentence = Sentence(tree_str)
    semantics = {}

    #print 
    #print sentence.getLocation()
    semantic_args = []
    try:
        elem_trees = sentence.getElemTrees()
        surface = " ".join([elem_tree.getTerminal() for elem_tree in elem_trees])
        palocs = {elem.getPASLoc() for elem in elem_trees}
        annotations = [propbank[paloc] for paloc in palocs if paloc in propbank]

        for annotation in annotations:
            pred = annotation.getRoleSetId()
            pastruct = annotation.getPAStruct()
            for arg in pastruct.getArgs():
                arg_label = arg.arg_label
                wordspans = sorted(arg.getLocation().getAllWordSpans())
                frags = [[(et.getPOS(), et.getTerminal()) for et in ws.getSubTree(sentence).getDominatedElemTrees()] for ws in wordspans if sentence.getSubTree(ws) is not None]
                if arg_label.isNumbered():
                    #print (pred, arg_label, wordspans, frags)
                    semantic_args += frags
        output_file.write(str(semantic_args) + "\n")
        '''
            if elem.getPASLoc() in propbank:
                annotation = propbank[elem.getPASLoc()]
                pred = annotation.getRoleSet()
                pastruct = annotation.getPAStruct()
                print (elem, elem.getSpan().toString(), [a for a in pastruct.getArgs()], [s.toString() for a in pastruct.getArgs() for s in a.getLocation().getAllWordSpans()])
                matching_args = [a for a in pastruct.getArgs() if elem.getSpan().toString() in [s.toString() for s in a.getLocation().getAllWordSpans()]]
                if len(matching_args) > 0:
                    label = matching_args[0].arg_label
                    if label.isNumbered():
                        print label, pred, elem.getSpan()
        '''

    except SkippedSentenceException:
        return None

if __name__ == "__main__":
    """This file is intended to be run from print_all_trees.sh """
    if sys.argv[3] == 'print_trees':
        tree_function = print_tree
    elif sys.argv[3] == 'print_args':
        tree_function = generate_semantics
    else:
        print("usage: jython print_generalized_trees.py <section_num> <output_filename> [print_trees | print_args]")
    process(sys.argv[1], output_filename=sys.argv[2], tree_function=tree_function)
