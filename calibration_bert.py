# run_ner.py
# ----------------------------------------
# Fully consistent training/evaluation script
# aligned with the updated utils_ner.py
# ----------------------------------------
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from seqeval.metrics import f1_score, precision_score, recall_score
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

from utils_ner import NerDataset, get_labels, Split

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path or identifier for pre-trained model"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast: bool = field(default=True)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataTrainingArguments:
    data_dir: str = field(metadata={"help": "Directory with train/dev/test files"})
    labels: Optional[str] = field(default=None)
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

def align_predictions_word_level(predictions, label_ids, word_ids_list, label_map):
    preds_argmax = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds_argmax.shape

    pred_labels, gold_labels = [], []
    for i in range(batch_size):
        pred, gold, word_ids = preds_argmax[i], label_ids[i], word_ids_list[i]
        sample_pred, sample_gold = [], []
        prev_word = None

        for idx in range(seq_len):
            word_idx = word_ids[idx]
            if word_idx is None or gold[idx] == -100:
                continue

            if word_idx != prev_word:
                sample_pred.append(label_map[pred[idx]])
                sample_gold.append(label_map[gold[idx]])
            prev_word = word_idx

        pred_labels.append(sample_pred)
        gold_labels.append(sample_gold)

    return pred_labels, gold_labels

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

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

    train_dataset = NerDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.train,
    ) if training_args.do_train else None

    eval_dataset = NerDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
    ) if training_args.do_eval else None

    test_dataset = NerDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        labels=labels,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.test,
    ) if training_args.do_predict else None

    def compute_metrics(p: EvalPrediction):
        preds, gold = align_predictions_word_level(
            p.predictions, p.label_ids, eval_dataset_word_ids, label_map
        )
        return {
            "precision": precision_score(gold, preds),
            "recall": recall_score(gold, preds),
            "f1": f1_score(gold, preds),
        }

    eval_dataset_word_ids = [item["word_ids"] for item in eval_dataset] if eval_dataset else None
    test_dataset_word_ids = [item["word_ids"] for item in test_dataset] if test_dataset else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        results = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key, value in results.items():
                writer.write(f"{key} = {value}\n")

    if training_args.do_predict:
        predictions, label_ids, _ = trainer.predict(test_dataset)
        preds, _ = align_predictions_word_level(predictions, label_ids, test_dataset_word_ids, label_map)
        
        test_examples = test_dataset.features
        output_pred_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        with open(output_pred_file, "w") as writer:
            for example, pred_labels in zip(test_examples, preds):
                for word, label in zip(example.words, pred_labels):
                    writer.write(f"{word}\t{label}\n")
                writer.write("\n")

if __name__ == "__main__":
    main()

