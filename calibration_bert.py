from typing import Tuple, Dict, List, Optional
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report
from dataclasses import dataclass, field
import logging
import os
import sys

from utils_ner import get_labels, NerDataset, Split, read_examples_from_file

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
    data_dir: str = field(metadata={"help": "Input data dir with .txt files for a CoNLL-2003-formatted task."})
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
    precision = precision_score(out_label_list, preds_list)
    recall = recall_score(out_label_list, preds_list)
    f1 = f1_score(out_label_list, preds_list)

    per_label_report = classification_report(out_label_list, preds_list, output_dict=True)
    logger.info("\n***** Per-Label Report *****")
    for lbl, vals in per_label_report.items():
        if isinstance(vals, dict):
            logger.info(f"Label: {lbl} | Precision={vals['precision']:.4f}, Recall={vals['recall']:.4f}, F1={vals['f1-score']:.4f}")

    return {"precision": precision, "recall": recall, "f1": f1}

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
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = eval_dataset = test_dataset = None
    if training_args.do_train:
        train_dataset = NerDataset(data_args.data_dir, tokenizer, labels, config.model_type, data_args.max_seq_length, data_args.overwrite_cache, Split.train)
    if training_args.do_eval:
        eval_dataset = NerDataset(data_args.data_dir, tokenizer, labels, config.model_type, data_args.max_seq_length, data_args.overwrite_cache, Split.dev)
    if training_args.do_predict:
        test_dataset = NerDataset(data_args.data_dir, tokenizer, labels, config.model_type, data_args.max_seq_length, data_args.overwrite_cache, Split.test)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        if trainer.is_world_process_zero():
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in result.items():
                    writer.write(f"{key} = {value}\n")

    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Test ***")
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write(f"{key} = {value}\n")

            preds_list, _ = align_predictions(predictions, label_ids, label_map)
            test_file_path = os.path.join(data_args.data_dir, "test.txt")
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            flat_preds = [p for pred in preds_list for p in pred]

            with open(test_file_path, "r", encoding="utf-8") as test_file, \
                 open(output_test_predictions_file, "w", encoding="utf-8") as writer:

                pred_idx = 0
                for line in test_file:
                    if line.strip() == "" or line.startswith("-DOCSTART-"):
                        writer.write(line)
                    else:
                        token = line.strip().split()[0]
                        if pred_idx < len(flat_preds):
                            pred_tag = flat_preds[pred_idx]
                            pred_idx += 1
                        else:
                            logger.warning("No prediction for token '%s'. Defaulting to 'O'.", token)
                            pred_tag = "O"
                        writer.write(f"{token}\t{pred_tag}\n")

def _mp_fn(index):
    main()

if __name__ == "__main__":
    main()

