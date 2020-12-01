# coding=utf-8
"""
Concatenate [CLS] and two entity markers for finetuning BERT model
"""
# coding=utf-8
import os
import argparse
import random
from collections import Counter
import numpy as np
from tqdm.auto import tqdm, trange
from sklearn import metrics
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    AdamW,
    BertTokenizer,
    BertConfig,
    BertModel,
    BertPreTrainedModel
)

from relex.datautils import load_relex_samples, load_id2label, create_sequence_with_markers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoolerLayer(nn.Module):
    
    def __init__(self, config):
        super(PoolerLayer, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, token_tensor):
        """

        Args:
            hidden_states (torch.FloatTensor): last hidden states of BERT for tokens
                                               size = (batch_size, hidden_size)
        Returns:
            pooled_output: Pool outputs
                           shape = (batch_size, hidden_size)
        """
        pooled_output = self.dense(token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForRelationClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForRelationClassification, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(3 * config.hidden_size, self.config.num_labels)
        self.init_weights()
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                e1_ids=None,
                e2_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]  # last hidden states of BERT for each token
        batch_size, _, hidden_size = sequence_output.size()
        
        # Put e1_token_tensor and e2_token_tensor to cuda if available
        e1_token_tensor = torch.zeros(batch_size, hidden_size, device=device)
        e2_token_tensor = torch.zeros(batch_size, hidden_size, device=device)
        
        for i in range(e1_ids.shape[0]):
            e1_token_tensor[i] = sequence_output[i, e1_ids[i]]
            e2_token_tensor[i] = sequence_output[i, e2_ids[i]]
        
        pooled_output = outputs[1]
        
        # Concatenate e1_token_tensor and e2_token_tensor
        pooled_output = torch.cat((pooled_output, e1_token_tensor, e2_token_tensor), 1)  # (batch_size, 3*hidden_size)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        
        return outputs  # (loss), logits, (hidden_states), (attentions)


def prepare_data(samples, labels, tokenizer, e1_start_token='[E1]', e1_end_token='[/E1]',
                 e2_start_token='[E2]', e2_end_token='[/E2]', maxlen=256):
    sequences = [create_sequence_with_markers(s, e1_start_token, e1_end_token,
                                              e2_start_token, e2_end_token) for s in samples]
    print("** First sequence with markers:", sequences[0])
    encoded_rs = tokenizer.batch_encode_plus(sequences, return_tensors='pt',
                                             padding=True,
                                             pad_to_max_length=True,
                                             max_length=maxlen)
    input_ids = encoded_rs["input_ids"]
    attention_mask = encoded_rs["attention_mask"]
    labels = torch.LongTensor(labels)
    
    # Calculate the start indexes of e1 and e2 entities
    e1_marker_idx, e2_marker_idx = tokenizer.convert_tokens_to_ids([e1_start_token, e2_start_token])
    
    e1_ids = [np.where(ids.numpy() == e1_marker_idx)[0][0] for ids in input_ids]
    e2_ids = [np.where(ids.numpy() == e2_marker_idx)[0][0] for ids in input_ids]
    
    e1_ids = torch.LongTensor(e1_ids)
    e2_ids = torch.LongTensor(e2_ids)
    
    dataset = TensorDataset(input_ids, e1_ids, e2_ids, attention_mask, labels)
    
    return dataset


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, model, tokenizer, id2label, train_dataset, validation_dataset):
    labels = [lb for lb in sorted(id2label.keys()) if id2label[lb] != 'OTHER']
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    
    val_sampler = SequentialSampler(validation_dataset)
    val_dataloader = DataLoader(validation_dataset, sampler=val_sampler, batch_size=args.eval_batch_size)
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.eps)
    
    print("***** Running training *****")
    print("  Num examples = %d" % len(train_dataset))
    print("  Num Epochs = %d" % args.epochs)
    
    set_seed()
    for epoch in trange(args.epochs, desc="Epoch"):
        # Training
        
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()
        
        # Tracking variables
        tr_loss, nb_tr_steps = 0, 0
        
        # Train the data for one epoch
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'e1_ids': batch[1],
                      'e2_ids': batch[2],
                      'attention_mask': batch[3],
                      'labels': batch[4]}
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_steps += 1
        
        print("Epoch {}, Train loss: {}".format(epoch + 1, tr_loss / nb_tr_steps))
        
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        
        # Tracking variables
        eval_loss, nb_eval_steps = 0, 0
        predictions, true_labels = [], []
        
        # Evaluate data for one epoch
        for batch in val_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(args.device) for t in batch)
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'e1_ids': batch[1],
                          'e2_ids': batch[2],
                          'attention_mask': batch[3],
                          'labels': batch[4]}
                b_labels = inputs['labels']
                # Forward pass, calculate logit predictions
                outputs = model(**inputs)
                loss, logits = outputs[:2]
            eval_loss += loss
            nb_eval_steps += 1
            
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)
        
        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        flat_true_labels = [item for sublist in true_labels for item in sublist]
        
        macro_f1 = metrics.f1_score(flat_true_labels, flat_predictions, labels=labels, average='macro')
        micro_f1 = metrics.f1_score(flat_true_labels, flat_predictions, labels=labels, average='micro')
        print(f"Epoch {epoch + 1}, Validation loss: {eval_loss / nb_eval_steps}, "
              f"Validation Accuracy: {metrics.accuracy_score(flat_true_labels, flat_predictions)}, "
              f"Validation Macro F1: {macro_f1}, "
              f"Validation Micro F1: {micro_f1}"
              )
        checkpoint_prefix = "checkpoint"
        output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, epoch + 1))
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
        with open(os.path.join(output_dir, "eval_results.txt"), "w") as fo:
            print(f"**** Epoch {epoch + 1} ****\n"
                  f"Validation loss: {eval_loss / nb_eval_steps}\n"
                  f"Validation Accuracy: {metrics.accuracy_score(flat_true_labels, flat_predictions)}\n"
                  f"Validation Macro F1: {macro_f1}\n"
                  f"Validation Micro F1: {micro_f1}\n",
                  file=fo,
                  )
        print("Saving model checkpoint to %s" % output_dir)


def evaluate(args, model, id2label, valid_dataset):
    labels = [lb for lb in sorted(id2label.keys()) if id2label[lb] != 'OTHER']
    text_labels = [id2label[lb] for lb in labels]
    
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler,
                                  batch_size=args.eval_batch_size)
    
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    predictions, true_labels = [], []
    
    eval_epoch_iterator = tqdm(valid_dataloader, desc="Evaluation")
    for batch in eval_epoch_iterator:
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'e1_ids': batch[1],
                      'e2_ids': batch[2],
                      'attention_mask': batch[3],
                      'labels': batch[4]
                      }
            b_labels = inputs['labels']
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
        
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        predictions.append(logits)
        true_labels.append(label_ids)
        nb_eval_steps += 1
        eval_loss += tmp_eval_loss.item()
    
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    
    true_labels = [id2label[i] for i in flat_true_labels]
    predictions = [id2label[i] for i in flat_predictions]
    
    macro_f1 = metrics.f1_score(flat_true_labels, flat_predictions, labels=labels, average='macro')
    micro_f1 = metrics.f1_score(flat_true_labels, flat_predictions, labels=labels, average='micro')
    print(f"Validation loss: {eval_loss / nb_eval_steps}, "
          f"Validation Accuracy: {metrics.accuracy_score(flat_true_labels, flat_predictions)}, "
          f"Validation Macro F1: {macro_f1}, "
          f"Validation Micro F1: {micro_f1}"
          )
    print("**** Classification Report ****")
    print(metrics.classification_report(true_labels, predictions, labels=text_labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True, help="Path to model checkpoint")
    
    parser.add_argument("--train_data_file", type=str, required=True,
                        help="The input training data file (a text file).")
    # Other parameters
    parser.add_argument(
        "--eval_data_file", required=True,
        type=str,
        help="Path to validation data (a text file).",
    )
    parser.add_argument(
        "--id2label", type=str, required=True,
        help="Path to id2label file"
    )
    parser.add_argument("--model_name_or_path", type=str, default="FPTAI/vibert-base-cased",
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
        default=8,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--maxlen",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (default is 2e-5)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="eps value for AdamW"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs"
    )
    args = parser.parse_args()
    if args.model_name_or_path in ["FPTAI/vibert-base-cased", "bert-base-multilingual-cased",
                                   "NlpHUST/vibert4news-base-cased"]:
        args.do_lower_case = False
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_samples, train_labels = load_relex_samples(args.train_data_file)
    id2label = load_id2label(args.id2label)
    num_labels = len(id2label)
    
    print("Train label distribution:")
    print(f"** Total training sample: {len(train_samples)}")
    print("**", Counter(train_labels))
    print()
    
    valid_samples, valid_labels = load_relex_samples(args.eval_data_file)
    print("Validation label distribution:")
    print(f"** Total training sample: {len(valid_samples)}")
    print("**", Counter(valid_labels))
    print()
    
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=args.do_lower_case)
    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels)
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])
    
    train_dataset = prepare_data(train_samples, train_labels, tokenizer=tokenizer, maxlen=args.maxlen)
    valid_dataset = prepare_data(valid_samples, valid_labels, tokenizer=tokenizer, maxlen=args.maxlen)
    
    model = BertForRelationClassification.from_pretrained(args.model_name_or_path, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    train(args, model, tokenizer, id2label, train_dataset, valid_dataset)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    
    evaluate(args, model, id2label, valid_dataset)

