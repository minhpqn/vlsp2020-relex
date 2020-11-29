# coding=utf-8
"""
Test data conversion function
"""
import re
import os
import sys
import argparse
import json
from data_converter import convert_conll_to_json, read_annotated_result, load_WebAnno_data, text_normalize


parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["1", "2"], default=1, help="Choose the test mode")
parser.add_argument("--input_file", type=str, required=True,
                    help="Path to input file")
args = parser.parse_args()

# E.g., "./data/VLSP2020_RE_training_fixed/23355434.conll/CURATION_USER.tsv"
file_id_match = re.search(r"/(\d+)\.conll/(.+)\.tsv", args.input_file)
if not file_id_match:
    sys.exit(1)

file_id = file_id_match.group(1)
file_name = file_id_match.group(2)

output_file = os.path.join("./tmp", file_id + ".json")
txt_output_file = os.path.join("./tmp", file_id + '.txt')
cached_file = os.path.join("./preprocessed_data/VLSP2020_RE_training-cached/" + file_id + ".conll", file_name + ".json")

print("txt_output_file: %s" % txt_output_file)


def test_read_annotated_result():
    print("Test read_annotated_result()")
    text, anno_tokens, id2token, relations = load_WebAnno_data(args.input_file)
    text = text_normalize(text)
    print("Loading parsing result from file %s" % cached_file)
    with open(cached_file, "r") as fi:
        annotated_data = json.load(fi)
    sentences = read_annotated_result(annotated_data, text)
    
    with open(txt_output_file, "w") as fo:
        for s in sentences:
            print(s, file=fo)
            print(file=fo)
    
    for tok in anno_tokens:
        print(tok)
        print()
    print()
    for rel in relations:
        print(rel)


def test_convert_conll_to_json():
    sentences = convert_conll_to_json(args.input_file, output_file, cached_file)
    
    with open(txt_output_file, "w") as fo:
        for s in sentences:
            print(s, file=fo)
            print(file=fo)

    
if __name__ == "__main__":
    if int(args.mode) == 1:
        test_read_annotated_result()
    else:
        test_convert_conll_to_json()
    
    
    

