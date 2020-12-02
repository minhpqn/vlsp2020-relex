# coding=utf-8
# Developed by Pham Quang Nhat Minh
"""
Train BERT model for relation classification
Current code is based on HuggingFace transformers version 2.11.0
"""
import os
import argparse
import logging
import numpy as np
from tqdm.auto import tqdm, trange
from sklearn import metrics
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (
    AdamW,
    BertTokenizer,
    BertConfig,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from rbert.utils import init_logger, set_seed, ADDITIONAL_SPECIAL_TOKENS
from relex.datautils import load_relex_samples, load_id2label, create_sequence_with_markers

logger = logging.getLogger(__name__)


def evaluate(args, model, id2label, valid_dataset):
    labels = [lb for lb in sorted(id2label.keys()) if id2label[lb] != 'OTHER']

    eval_sampler = SequentialSampler(valid_dataset)
    eval_dataloader = DataLoader(valid_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation on validation dataset *****")
    logger.info("  Num examples = %d", len(valid_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {"loss": eval_loss}
    preds = np.argmax(preds, axis=1)

    macro_f1 = metrics.f1_score(out_label_ids, preds, labels=labels, average='macro')
    micro_f1 = metrics.f1_score(out_label_ids, preds, labels=labels, average='micro')
    acc = metrics.accuracy_score(out_label_ids, preds)

    result = {
        'acc': acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1
    }
    results.update(result)

    logger.info("***** Eval results *****")
    for key in sorted(results.keys()):
        logger.info("  {} = {:.4f}".format(key, results[key]))

    if args.print_report:
        true_labels = [id2label[i] for i in out_label_ids]
        predictions = [id2label[i] for i in preds]
        text_labels = [id2label[lb] for lb in labels]
        logger.info("**** Classification Report ****")
        print(metrics.classification_report(true_labels, predictions, labels=text_labels, digits=4))

    return results
    

def train(args, model, tokenizer, id2label, train_dataset, valid_dataset=None):
    labels = [lb for lb in sorted(id2label.keys()) if id2label[lb] != 'OTHER']
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  batch_size=args.eval_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        )
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.adam_epsilon,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    evaluate(args, model, id2label, valid_dataset)
                
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_model(args, model, tokenizer)
                
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        return global_step, tr_loss / global_step


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretraiend(args.output_dir)
    # Save training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    logger.info("Saving model checkpoint to %s", args.output_dir)

def prepare_data(samples, labels, tokenizer, maxlen=256):
    sequences = [create_sequence_with_markers(s) for s in samples]
    logger.info("** First sequence with markers: {}".format(sequences[0]))
    encoded_rs = tokenizer.batch_encode_plus(sequences, return_tensors='pt',
                                             padding=True,
                                             pad_to_max_length=True,
                                             max_length=maxlen)
    input_ids = encoded_rs["input_ids"]
    attention_mask = encoded_rs["attention_mask"]
    labels = torch.LongTensor(labels)
    dataset = TensorDataset(input_ids, attention_mask, labels)
    
    return dataset


def main(args):
    init_logger()
    set_seed(args)
    
    logger.info("%s" % args)
    os.makedirs(args.output_dir, exist_ok=True)
    train_samples, train_labels = load_relex_samples(args.train_data_file)
    id2label = load_id2label(args.id2label)
    num_labels = len(id2label)
    
    valid_samples, valid_labels = load_relex_samples(args.eval_data_file)
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=False)
    tokenizer.add_tokens(ADDITIONAL_SPECIAL_TOKENS)
    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels,
                                        finetuning_task="VLSP2020-Relex",
                                        id2label={str(i): label for i, label in id2label.items()},
                                        label2id={label: i for i, label in id2label.items()},
                                        )
    train_dataset = prepare_data(train_samples, train_labels, tokenizer=tokenizer, maxlen=args.maxlen)
    valid_dataset = prepare_data(valid_samples, valid_labels, tokenizer=tokenizer, maxlen=args.maxlen)
    
    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                          config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    train(args, model, tokenizer, id2label, train_dataset, valid_dataset)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    evaluate(args, model, id2label, valid_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to model checkpoint")
    
    parser.add_argument("--train_data_file", type=str,
                        default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/train.txt",
                        help="The input training data file (a text file).")
    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/dev.txt",
        type=str,
        help="Path to validation data (a text file).",
    )
    parser.add_argument(
        "--id2label", type=str,
        default="./preprocessed_data/VLSP2020_RE_SemEvalFormat/id2label.txt",
        help="Path to id2label file"
    )
    parser.add_argument("--model_name_or_path", type=str, default="FPTAI/vibert-base-cased",
                        choices=["FPTAI/vibert-base-cased", "bert-base-multilingual-cased", "NlpHUST/vibert4news-base-cased"],
                        help="Model name or path to BERT model")
    parser.add_argument("--do_lower_case", action="store_true", help="Whether to lower case texts")
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=384,
        help="Maximum sequence length"
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument("--seed", type=int, default=77, help="Random seed")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate (default is 2e-5)"
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=10.0,
        help="Training epochs"
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=250, help="Log every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=250,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main(args)


