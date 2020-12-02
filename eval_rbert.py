# coding=utf-8
"""
Evaluate R-BERT model on run_rbert.py
"""
import os
import argparse
from collections import Counter
import torch
from transformers import BertTokenizer
from relex.datautils import load_relex_samples, load_id2label
from run_rbert import RBERT, prepare_data, evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Model name or path to BERT model")
    parser.add_argument(
        "--eval_data_file",
        required=True, type=str,
        help="Path to validation data (a text file).",
    )
    args = parser.parse_args()
    training_args = torch.load(os.path.join(args.model_path, "training_args.bin"))
    
    valid_samples, valid_labels = load_relex_samples(args.eval_data_file)
    print("Validation label distribution:")
    print(f"** Total validation samples: {len(valid_samples)}")
    print("**", Counter(valid_labels))
    print()
    tokenizer = BertTokenizer.from_pretrained(args.model_path,
                                              do_lower_case=training_args.do_lower_case)
    valid_dataset = prepare_data(valid_samples, valid_labels, tokenizer=tokenizer, maxlen=training_args.maxlen)
    model = RBERT.from_pretrained(args.model_path)
    model = model.to(training_args.device)
    id2label = load_id2label(training_args.id2label)
    evaluate(training_args, model, id2label, valid_dataset)

