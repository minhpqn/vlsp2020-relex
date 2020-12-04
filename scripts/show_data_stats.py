# coding=utf-8
"""
Print statistics for relation types
"""
import os
import re
import argparse
from collections import Counter


def load_id2label(file_path):
    """Load id2label from file id2label.txt
    """
    id2label = {}
    with open(file_path, 'r') as fi:
        for line in fi:
            line = line.strip()
            if line == "":
                continue
            i, lb = line.split("\t")
            id2label[int(i)] = lb
    return id2label


def show_stats_on_raw_data(input_dir):
    relation_types = []
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
               and re.search(r"\.conll", d)]
    print("    Number of sub-directories: {}".format(len(subdirs)))
    
    for subdir in subdirs:
        subdir_ = os.path.join(input_dir, subdir)
        files = [f for f in os.listdir(subdir_) if re.search(r"\.tsv", f)]
        for file in files:
            file_path = os.path.join(subdir_, file)
            with open(file_path, 'r') as fi:
                i = 0
                for line in fi:
                    i += 1
                    line = line.rstrip()
                    if line == "":
                        continue
                    if re.search(r"#Text=(.+)$", line):
                        continue
                    elif re.search(r"^1-\d+", line):
                        fields = line.split("\t")
                        if len(fields) > 6:
                            rel_types = fields[5].split("|")
                            relation_types.extend([rel for rel in  rel_types if rel != "_"])
    print("*** Relation type counts: {}".format(Counter(relation_types)))


def show_stats_on_preprocessed_data(input_file, id2label):
    relation_types = []
    total_samples = 0
    with open(input_file, 'r') as f:
        for line in f:
            total_samples += 1
            fields = line.split("\t")
            lb = id2label[int(fields[0])]
            relation_types.append(lb)
    print("*** Number of samples: {}".format(total_samples))
    print("*** Relation type counts: {}".format(Counter(relation_types)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str,
                        default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/train.txt",
                        help="Path to train data file in semeval format")
    parser.add_argument("--dev_file", type=str,
                        default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/dev.txt",
                        help="Path to dev data file in semeval format")
    parser.add_argument("--id2label", type=str, default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/id2label.txt",
                        help="Path to id2label file")
    parser.add_argument("--train_dir", type=str, default="./data/VLSP2020_RE_training_fixed",
                        help="Path to train data dir")
    parser.add_argument("--dev_dir", type=str, default="./data/VLSP2020_RE_dev_fixed",
                        help="Path to dev data dir")
    args = parser.parse_args()
    
    if args.train_dir:
        print("****** Data statistics for train data ******")
        print("Train data dir: {}".format(args.train_dir))
        show_stats_on_raw_data(args.train_dir)
        print()
    
    if args.dev_dir:
        print("****** Data statistics for dev data ******")
        print("Dev data dir: {}".format(args.dev_dir))
        show_stats_on_raw_data(args.dev_dir)
        print()
    
    id2label = None
    if args.id2label:
        id2label = load_id2label(args.id2label)
    
    if args.train_file:
        print("****** Data statstics for train file ******")
        print("Train file: {}".format(args.train_file))
        show_stats_on_preprocessed_data(args.train_file, id2label)
        print()
    
    if args.train_file:
        print("****** Data statstics for dev file ******")
        print("Dev file: {}".format(args.dev_file))
        show_stats_on_preprocessed_data(args.dev_file, id2label)
        print()

