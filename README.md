# Relation Extraction at VLSP 2020

(C) Copyright by Pham Quang Nhat Minh

Official code for the paper [An Empirical Study of Using Pre-trained BERT Models for Vietnamese Relation Extraction Task at VLSP 2020](https://arxiv.org/pdf/2012.10275.pdf), VLSP 2020. In the paper, we used BERT-based models for Vietnamese Relation Extraction. Our system ranked second in the VLSP 2020 shared task.

Details of the models and experimental results can be found in the following paper.

    @inproceedings{pham2020empirical,
    title={An Empirical Study of Using Pre-trained BERT Models for Vietnamese Relation Extraction Task at VLSP 2020},
    author={Pham, Minh Quang Nhat},
    booktitle={Proceedings of the 7th International Workshop on Vietnamese Language and Speech Processing},
    pages={13--18},
    year={2020}
    }

## Requirements

We tested the code with:

- Python 3.7.10
- Pytorch 1.8.1
- Hugging Face Transformers 4.6.0
- googledrivedownloader (pip install googledrivedownloader).
- Modified version of pyvi from the [forked repo](https://github.com/minhpqn/pyvi) for word segmentation with syllablized texts.

## Evaluation Results

**NOTE:** We added evaluation results with PhoBERT pre-trained model. These results were not reported in the paper. Experiments with PhoBERT is quite tricky because we need to fixed word segmentation errors in which an entity mention contains syllables of another words.

We did not submit results on test-set with PhoBERT for official evaluation, so we only have evaluation results on development set with PhoBERT.

| Model    | Pre-trained Model | MACRO F1   | MICRO F1 |
|----------|-------------------|------------|----------|
| R-BERT   | NlpHUST/vibert4news | 0.6392 | 0.7092 |
| R-BERT   | vinai/phobert-base  | 0.6635 | 0.7228 |
| R-BERT   | FPTAI/vibert  | 0.596 | 0.6736 |
| BERT-ES  | NlpHUST/vibert4news | 0.6439 | 0.7101 |
| BERT-ES  | vinai/phobert-base  | 0.6651 | 0.7262 |
| BERT-ES  | FPTAI/vibert  | 0.596 | 0.6736 |
| Ensemble | NlpHUST/vibert4news | 0.6412 | 0.7108 |
| Ensemble | vinai/phobert-base  | **0.6687** | **0.7299** |
| Ensemble | FPTAI/vibert  | 0.6029 | 0.6851 |

On the development set, ensemble model + PhoBERT obtained the best results among models.

## Data Preparation

From the original Vietnamese relation extraction dataset provided by VLSP 2020 organizers, we converted to
SemEval format. In SemEval format, a file is a list of lines and each line is tab delimitted with following information.

    <label> <entity1_start_index> <entity1_end_index> <entity2_start_index> <entity2_end_index> <entity1_type>  <entity2_type> <tokenized_sentence>

An example of a sample

    3	13	13	17	18	LOCATION	LOCATION	Đây là sự kiện quan trọng đối với ngành chăn nuôi gà vùng ĐBSCL nói riêng , Việt Nam nói chung .

In order to do that, we need to use VnCoreNLP for sentence segmentation and tokenization.

    python data_converter.py --overwrite_output_dir --input_dir ./data/VLSP2020_RE_training --output_dir preprocessed_data/VLSP2020_RE_training

    python data_converter.py --overwrite_output_dir --input_dir ./data/VLSP2020_RE_dev --output_dir preprocessed_data/VLSP2020_RE_dev

The directory `preprocessed_data/VLSP2020_RE_training` is then used to generate data in SemEval format.

Use the script `semeval_converter.py` for converting data into SemEval format.

    python semeval_converter.py  --input_dir ./preprocessed_data/VLSP2020_RE_training --output_dir ./data/VLSP2020_RE_SemEvalFormat

    python semeval_converter.py  --input_dir ./preprocessed_data/VLSP2020_RE_dev --output_dir ./data/VLSP2020_RE_SemEvalFormat

Now, directory `./data/VLSP2020_RE_SemEvalFormat` contains necessary files for training/evaluation.

- train.txt
- dev.txt
- id2label.txt

`id2label.txt` contains the mapping from indexes to labels.

```
0	AFFILIATION
1	LOCATED
2	OTHER
3	PART – WHOLE
4	PERSONAL - SOCIAL
```

Transforming data into PhoBERT is a bit tricky because we need to fix entity boundary errors. You have to install pyvi package from [forked repo](https://github.com/minhpqn/pyvi) here. Then run the script `prepare_data4phobert.py` as follows.

```
python prepare_data4phobert.py --input_file ./data/VLSP2020_RE_SemEvalFormat/dev.txt --id2label ./data/VLSP2020_RE_SemEvalFormat/id2label.txt

python prepare_data4phobert.py --input_file ./data/VLSP2020_RE_SemEvalFormat/train.txt --id2label ./data/VLSP2020_RE_SemEvalFormat/id2label.txt
```

**NOTE**: You need to apply for VLSP 2020 - Relex dataset. See [VLSP 2020 homepage](https://vlsp.org.vn/) for instructions.

### Downloading cached feature files

**Due to the copyright of VLSP 2020 Relex data**, we could not share data. However, to reproduce the results in the paper, we can use cached features extracted from the original data. Download cached features using the following command.

    sh download_data.sh

## Training and Evaluation

### BERT-ES (BERT Entity Markers + Entity Starts)

With `NlpHUST/vibert4news-base-cased` pre-trained model.

```
python run_bert_em.py --model_dir ./models/original_train_dev/bert_em_es_bert4news_maxlen_384_epochs_10 \
                      --model_name_or_path NlpHUST/vibert4news-base-cased \
                      --train_data_file ./data/cached_features/train.txt  \
                      --eval_data_file ./data/cached_features/dev.txt \
                      --id2label ./data/cached_features/id2label.txt  \
                      --train_batch_size 8 --gradient_accumulation_steps 2 \
                      --save_steps 1000 --logging_steps 1000 \
                      --model_type es --do_train --do_eval
```

With `FPTAI/vibert-base-cased` pre-trained model.

```
python run_bert_em.py --model_dir ./models/original_train_dev/bert_em_es_fptbert_maxlen_384_epochs_10 \
                      --model_name_or_path FPTAI/vibert-base-cased \
                      --train_data_file ./data/cached_features/train.txt  \
                      --eval_data_file ./data/cached_features/dev.txt \
                      --id2label ./data/cached_features/id2label.txt  \
                      --train_batch_size 8 --gradient_accumulation_steps 2 \
                      --save_steps 1000 --logging_steps 1000 \
                      --model_type es --do_train --do_eval
```

With `vinai/phobert-case` pre-trained models.

```
python run_phobert_em.py --model_dir ./models/original_train_dev/phobert_em_es_base_maxlen_384_epochs_10 \
                         --train_data_file ./data/cached_features/train-phobert.txt \
                         --eval_data_file ./data/cached_features/dev-phobert.txt \
                         --id2label ./data/cached_features/id2label.txt \
                         --train_batch_size 8 --gradient_accumulation_steps 2 \
                         --save_steps 1000 --logging_steps 1000 \
                         --model_type es --do_train --do_eval
```

### R-BERT

With `NlpHUST/vibert4news-base-cased` pre-trained model.

```
python run_rbert.py --model_dir ./models/original_train_dev/bert_em_es_bert4news_maxlen_384_epochs_10 \
                    --model_name_or_path NlpHUST/vibert4news-base-cased \
                    --train_data_file ./data/cached_features/train.txt \
                    --eval_data_file ./data/cached_features/dev.txt \
                    --id2label ./data/cached_features/id2label.txt \
                    --train_batch_size 8 --gradient_accumulation_steps 2 \
                    --save_steps 1000 --logging_steps 1000 \
                    --do_train --do_eval
```

With `FPTAI/vibert-base-cased` pre-trained model.

```
python run_rbert.py --model_dir ./models/original_train_dev/rbert_fptbert_maxlen_384_epochs_10 \
                    --model_name_or_path FPTAI/vibert-base-cased \
                    --train_data_file ./data/cached_features/train.txt \
                    --eval_data_file ./data/cached_features/dev.txt \
                    --id2label ./data/cached_features/id2label.txt \
                    --train_batch_size 8 --gradient_accumulation_steps 2 \
                    --save_steps 1000 --logging_steps 1000 \
                    --do_train --do_eval
```

With `vinai/phobert-base`

```
python run_phobert_rbert.py --model_dir ./models/original_train_dev/phobert_rbert_base_maxlen_384_epochs_10 \
                            --train_data_file ./data/cached_features/train-phobert.txt \
                            --eval_data_file ./data/cached_features/dev-phobert.txt \
                            --id2label ./data/cached_features/id2label.txt \
                            --train_batch_size 8 --gradient_accumulation_steps 2 \
                            --save_steps 1000 --logging_steps 1000 \
                            --do_train --do_eval
```

### Ensemble models

With `NlpHUST/vibert4news-base-cased` pre-trained model.

    python run_ensemble.py --input_file ./data/cached_features/dev.txt

With `NlpHUST/vibert4news-base-cased` pre-trained model.

    python run_ensemble_fptbert.py --input_file ./data/cached_features/dev.txt

With `vinai/phobert-base`

    python python run_ensemble_phobert.py --input_file ./data/cached_features/dev-phobert.txt


