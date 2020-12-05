# coding=utf-8
"""
Remove labels from files in a directory
"""
import os
import re
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Path to input directory")
    parser.add_argument("output_dir", help="Path to output directory")
    args = parser.parse_args()
    
    base_dirname = os.path.basename(args.input_dir)
    base_dirname = base_dirname.split("/")[0]
    output_dir = os.path.join(args.output_dir, base_dirname)
    os.makedirs(output_dir, exist_ok=True)
    
    print("{} --> {}".format(args.input_dir, output_dir))
    
    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    for file in files:
        input_path = os.path.join(args.input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        with open(output_path, "w") as fo:
            with open(input_path, "r") as fi:
                for line in fi:
                    if re.search(r"^1-\d+", line):
                        fields = line.split("\t")
                        assert len(fields) >= 5, "{}\t{}".format(line.rstrip(), input_path)
                        new_fields = fields[:5]
                        new_fields.append("_")
                        new_fields.append("_")
                        print("{}".format("\t".join(new_fields)), file=fo)
                    else:
                        fo.write("{}".format(line))

