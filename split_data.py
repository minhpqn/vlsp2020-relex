# coding=utf-8
"""
Split train+dev dataset into new train/dev data
We keep just 100 documents in dev as the new dev data
"""
import re
import os
import argparse
from distutils.dir_util import copy_tree
import random
random.seed(77)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for new train/dev data")
    parser.add_argument("--dev_size", type=int, default=100,
                        help="Keep 100 dirs as new dev data")
    parser.add_argument("--train_dir", type=str, default="./data/VLSP2020_RE_training_fixed",
                        help="Path to train data dir")
    parser.add_argument("--dev_dir", type=str, default="./data/VLSP2020_RE_dev_fixed",
                        help="Path to dev data dir")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    new_train_dir = os.path.join(args.output_dir, "VLSP2020_RE_new_train")
    new_dev_dir = os.path.join(args.output_dir, "VLSP2020_RE_new_dev")
    
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_dev_dir, exist_ok=True)

    train_subdirs = [d for d in os.listdir(args.train_dir) if os.path.isdir(os.path.join(args.train_dir, d))
                       and re.search(r"\.conll", d)]
    
    for subdir in train_subdirs:
        input_dir = os.path.join(args.train_dir, subdir)
        output_dir = os.path.join(new_train_dir, subdir)
        copy_tree(input_dir, output_dir)

    dev_subdirs = [d for d in os.listdir(args.dev_dir) if os.path.isdir(os.path.join(args.dev_dir, d))
                     and re.search(r"\.conll", d)]
    dev_subdirs = sorted(dev_subdirs)
    random.shuffle(dev_subdirs)
    
    for subdir in dev_subdirs[0:args.dev_size]:
        input_dir = os.path.join(args.dev_dir, subdir)
        output_dir = os.path.join(new_dev_dir, subdir)
        copy_tree(input_dir, output_dir)
    
    for subdir in dev_subdirs[args.dev_size:]:
        input_dir = os.path.join(args.dev_dir, subdir)
        output_dir = os.path.join(new_train_dir, subdir)
        copy_tree(input_dir, output_dir)
    

