# coding=utf-8
"""
Implementation of R-BERT model
"""
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


def entity_average(sequence_output, start_ids, end_ids, pooled_output):
    batch_size, _, hidden_size = sequence_output.size()
    
    token_tensor = torch.zeros(batch_size, hidden_size, device=device)
    
    for i in range(start_ids.shape[0]):
        if start_ids[i] == -1:
            token_tensor[i] = pooled_output[i].squeeze(0)
        else:
            k = start_ids[i] + 1
            m = end_ids[i] - 1
            entity_length = m - k + 1
            token_tensor[i] = torch.sum(sequence_output[i, k:m + 1]) / entity_length
    
    return token_tensor


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERT(BertPreTrainedModel):
    
    def __init__(self, config, dropout_rate):
        super(RBERT, self).__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        
        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, dropout_rate)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.label_classifier = FCLayer(
            config.hidden_size * 3,
            config.num_labels,
            dropout_rate,
            use_activation=False,
        )
        self.init_weights()

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]
    
        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector
    
    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                e1_mask=None,
                e2_mask=None,
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
        pooled_output = outputs[1]
    
        # Average of tokens between [E1] and [/E1]
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)
    
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
    all_input_ids = encoded_rs["input_ids"]
    all_attention_mask = encoded_rs["attention_mask"]
    all_labels = torch.LongTensor(labels)
    
    # Calculate the start indexes of e1 and e2 entities
    e1_start_marker_idx, e1_end_marker_idx, e2_start_marker_idx, e2_end_marker_idx = \
        tokenizer.convert_tokens_to_ids([e1_start_token, e1_end_token, e2_start_token, e2_end_token])
    
    all_e1_mask = []
    all_e2_mask = []
    
    for input_ids in all_input_ids:
        input_ids = input_ids.numpy().tolist()
        e1_start_id = input_ids.index(e1_start_marker_idx)
        e1_end_id = input_ids.index(e1_end_marker_idx)
        e2_start_id = input_ids.index(e2_start_marker_idx)
        e2_end_id = input_ids.index(e2_end_marker_idx)
        
        e1_mask = [0] * len(input_ids)
        e2_mask = [0] * len(input_ids)
        for i in range(e1_start_id, e1_end_id + 1):
            e1_mask[i] = 1
        for i in range(e2_start_id, e2_end_id + 1):
            e2_mask[i] = 1
        
        all_e1_mask.append(e1_mask)
        all_e2_mask.append(e2_mask)
    
    all_e1_mask = torch.LongTensor(all_e1_mask)
    all_e2_mask = torch.LongTensor(all_e2_mask)
    
    dataset = TensorDataset(all_input_ids, all_e1_mask, all_e2_mask, all_attention_mask, all_labels)
    
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
                      'e1_mask': batch[1],
                      'e2_mask': batch[2],
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
                          'e1_mask': batch[1],
                          'e2_mask': batch[2],
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
        output_dir = os.path.join(args.output_dir, "{}-{}_{:.3f}_{:.3f}".format(checkpoint_prefix,
                                                                                epoch + 1, macro_f1, micro_f1))
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
                      'e1_mask': batch[1],
                      'e2_mask': batch[2],
                      'attention_mask': batch[3],
                      'labels': batch[4]}
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
    print(metrics.classification_report(true_labels, predictions, labels=text_labels, digits=4))
    
    
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
        "--dropout_rate",
        default=0.1,
        type=float,
        help="Dropout for fully-connected layers",
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
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_samples, train_labels = load_relex_samples(args.train_data_file, do_lower_case=args.do_lower_case)
    id2label = load_id2label(args.id2label)
    num_labels = len(id2label)
    
    print("Train label distribution:")
    print(f"** Total training sample: {len(train_samples)}")
    print("**", Counter(train_labels))
    print()
    
    valid_samples, valid_labels = load_relex_samples(args.eval_data_file, do_lower_case=args.do_lower_case)
    print("Validation label distribution:")
    print(f"** Total training sample: {len(valid_samples)}")
    print("**", Counter(valid_labels))
    print()

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              do_lower_case=False)
    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=num_labels)
    tokenizer.add_tokens(['[E1]', '[/E1]', '[E2]', '[/E2]'])

    train_dataset = prepare_data(train_samples, train_labels, tokenizer=tokenizer, maxlen=args.maxlen)
    valid_dataset = prepare_data(valid_samples, valid_labels, tokenizer=tokenizer, maxlen=args.maxlen)

    model = RBERT.from_pretrained(args.model_name_or_path, config=config, dropout_rate=args.dropout_rate)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    train(args, model, tokenizer, id2label, train_dataset, valid_dataset)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    evaluate(args, model, id2label, valid_dataset)