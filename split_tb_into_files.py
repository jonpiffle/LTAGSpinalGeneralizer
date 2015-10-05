import re, sys

def split_treebank(treebank_dir="/Users/piffle/Desktop/spinalapi/spinalapi/ltagtb/", output_dir='trees/'):
    filenames = [
        "derivation.sec0-1.v01",
        "derivation.train.v01",
        "derivation.sec22.v01",
        "derivation.test.v01",
        "derivation.devel.v01",
    ]
    tree_begin = "^\d+ \d+ \d+$"
    tree_end = "^\s$"

    for filename in filenames:
        with open(directory + filename) as f:
            tree = []
            for line in f:
                m_begin = re.search(tree_begin, line)
                m_end = re.search(tree_end, line)

                if m_begin:
                    tree = [line]
                elif m_end:
                    output_file = 'trees/' + tree[0].replace(" ", "_").strip() + ".txt"
                    with open(output_file, 'w') as f:
                        f.write(''.join(tree))
                    tree = []
                else:
                    tree.append(line)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        split_treebank()
    elif len(sys.argv) == 3:
        split_treebank(sys.argv[1], sys.argv[2])
    else:
        print('Usage: python split_tb_into_files.py path_to_treebank output_dir')
