#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedModel,
    PretrainedConfig
)
from transformers.modeling_outputs import TokenClassifierOutput

from seqeval.metrics import classification_report

from utils_ner import get_labels, NerDataset, Split

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast: bool = field(default=False)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataTrainingArguments:
    data_dir: str = field(metadata={"help": "Input data dir with .tsv files for NER"})
    labels: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, label_map: Dict[int, str]) -> Tuple[List[List[str]], List[List[str]]]:
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list

def compute_metrics(p: EvalPrediction) -> Dict[str, float]:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, label_map)
    report = classification_report(out_label_list, preds_list, output_dict=True, zero_division=0)
    metrics = {}
    for lbl, vals in report.items():
        if isinstance(vals, dict) and lbl not in ["micro avg", "macro avg", "weighted avg", "accuracy"]:
            metrics[f"{lbl}_precision"] = vals["precision"]
            metrics[f"{lbl}_recall"] = vals["recall"]
            metrics[f"{lbl}_f1"] = vals["f1-score"]
    if "macro avg" in report:
        metrics["overall_precision"] = report["macro avg"]["precision"]
        metrics["overall_recall"] = report["macro avg"]["recall"]
        metrics["overall_f1"] = report["macro avg"]["f1-score"]
    return metrics

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    global label_map
    labels = get_labels(data_args.labels)
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name or model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.do_predict:
        test_dataset = NerDataset(data_args.data_dir, tokenizer, labels, config.model_type,
                                  data_args.max_seq_length, data_args.overwrite_cache, Split.test)
    else:
        test_dataset = None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        compute_metrics=compute_metrics,
    )

    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Test ***")
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids, label_map)

        test_file_path = os.path.join(data_args.data_dir, "test.tsv")
        output_predictions_file = os.path.join(training_args.output_dir, "test_predictions.tsv")

        with open(test_file_path, "r", encoding="utf-8") as f, open(output_predictions_file, "w", encoding="utf-8") as writer:
            example_id = 0
            token_id = 0
            for line in f:
                striped = line.strip()
                if striped == "":
                    writer.write(".\tO\n")
                    continue
                splits = striped.split("\t")
                token = splits[0]
                if example_id < len(preds_list) and token_id < len(preds_list[example_id]):
                    pred_tag = preds_list[example_id][token_id]
                    token_id += 1
                else:
                    pred_tag = "O"
                writer.write(f"{token}\t{pred_tag}\n")
                if token_id == len(preds_list[example_id]):
                    example_id += 1
                    token_id = 0

if __name__ == "__main__":
    main()



