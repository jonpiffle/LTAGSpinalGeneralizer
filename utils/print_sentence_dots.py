import sys
sys.path.append('/Users/piffle/Documents/luna_workspace/spinal/bin')
from edu.upenn.cis.spinal import *
from java.lang import NullPointerException

propbank_path = '/Users/piffle/Desktop/spinalapi/spinalapi/prop-all.idx'
ltagtb_path = '/Users/piffle/Desktop/spinalapi/spinalapi/ltagtb/'

def ltagtb_filename(dir_num):
    dir_num = int(dir_num)
    if dir_num < 2:
        return "derivation.sec0-1.v01"
    elif dir_num < 22:
        return "derivation.train.v01"
    elif dir_num < 23:
        return "derivation.sec22.v01"
    elif dir_num < 24:
        return "derivation.test.v01"
    elif dir_num < 25:
        return "derivation.devel.v01"
    else:
        raise ValueError("dir_num is outside value range")


def filename_to_address(filename):
    """wsj/00/wsj_0001.mrg -> (0, 1)"""
    fid = filename.split(".")[0].split("_")[-1]
    dir_num = str(int(fid[:2]))
    file_num = str(int(fid[2:]))
    return dir_num, file_num

def print_pred(pred_key):
    with open(propbank_path) as f:
        for line in f: 
            chunks = line.split(" ")
            filename, sen_num, word_num, corp, pred = chunks[:5]

            if pred != pred_key:
                continue

            dir_num, file_num = filename_to_address(filename)
            ltagtb_file = ltagtb_path + ltagtb_filename(dir_num)
            print(dir_num, file_num, sen_num, ltagtb_file)
            GraphvizWalker.main([ltagtb_file, dir_num, file_num, sen_num])

if __name__ == '__main__':
    #command to convert all dots to pngs: dot *.dot -Tpng -O
    print_pred('join.01')
