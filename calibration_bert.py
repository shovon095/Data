# run_ner.py
# ----------------------------------------
# Minimal example of how to train and predict
# with the updated utils_ner.py
# ----------------------------------------
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
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

# Import from our updated utils_ner
from utils_ner import (
    get_labels,
    NerDataset,
    Split,
    InputFeatures,
    InputExample,
)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or huggingface.co model identifier"}
    )
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    use_fast: bool = field(default=True)
    cache_dir: Optional[str] = field(default=None)

@dataclass
class DataTrainingArguments:
    data_dir: str = field(metadata={"help": "The input data dir with train/dev/test .txt files"})
    labels: Optional[str] = field(default=None, metadata={"help": "Path to a file containing labels (one per line)"})
    max_seq_length: int = field(default=128)
    overwrite_cache: bool = field(default=False)

def align_predictions_word_level(
    predictions: np.ndarray,  # shape (n_samples, seq_len, num_labels)
    label_ids: np.ndarray,    # shape (n_samples, seq_len)
    word_ids_list: List[List[Optional[int]]],
    label_map: Dict[int, str],
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Reconstruct the predicted labels at the word level using the
    stored 'word_ids'.
    - predictions: raw logits from the model
    - label_ids: gold label IDs (with -100 for special tokens)
    - word_ids_list: each item is the word_ids for that example
    - label_map: maps label_id -> label string
    Returns:
      pred_word_labels, gold_word_labels
    """

    preds_argmax = np.argmax(predictions, axis=2)  # shape: (n_samples, seq_len)
    batch_size, seq_len = preds_argmax.shape

    pred_word_labels = []
    gold_word_labels = []

    for i in range(batch_size):
        pred_ids = preds_argmax[i]
        gold_ids = label_ids[i]
        word_ids = word_ids_list[i]

        # We'll build a list of final labels (one per original word)
        current_word_id = None
        current_pred_label = None
        current_gold_label = None

        sample_pred_labels = []
        sample_gold_labels = []

        for j in range(seq_len):
            w_id = word_ids[j]
            if w_id is None:
                # special token (CLS, SEP, PAD)
                continue
            if gold_ids[j] == -100:
                # subword label or something to ignore
                continue

            # This subword belongs to word index = w_id
            # If we see a "new" w_id, that means a new word
            if w_id != current_word_id:
                # finalize the old word
                if current_word_id is not None:
                    sample_pred_labels.append(label_map[current_pred_label])
                    sample_gold_labels.append(label_map[current_gold_label])

                current_word_id = w_id
                current_pred_label = pred_ids[j]
                current_gold_label = gold_ids[j]
            else:
                # same word: we could do majority-vote among subwords
                # or just keep the first subword's label
                # We'll do "keep the first" (so do nothing here)
                pass

        # finalize last word in the sequence
        if current_word_id is not None:
            sample_pred_labels.append(label_map[current_pred_label])
            sample_gold_labels.append(label_map[current_gold_label])

        pred_word_labels.append(sample_pred_labels)
        gold_word_labels.append(sample_gold_labels)

    return pred_word_labels, gold_word_labels


def compute_metrics(p: EvalPrediction, label_map: Dict[int, str], word_ids_for_eval) -> Dict[str, float]:
    """
    Called by Trainer to compute metrics. We realign
    predictions -> word-level, then compute F1 etc. with seqeval.
    """
    preds, out_label_ids = p.predictions, p.label_ids  # shapes: (n_samples, seq_len, num_labels), (n_samples, seq_len)
    pred_word_labels, gold_word_labels = align_predictions_word_level(
        preds,
        out_label_ids,
        word_ids_for_eval,
        label_map,
    )

    precision = precision_score(gold_word_labels, pred_word_labels)
    recall = recall_score(gold_word_labels, pred_word_labels)
    f1 = f1_score(gold_word_labels, pred_word_labels)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If passing one .json config file
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

    # 1) Prepare labels
    labels = get_labels(data_args.labels)  # read from file or default
    label_map = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # 2) Load config, tokenizer, model
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # 3) Load datasets
    train_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )
    test_dataset = (
        NerDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )
        if training_args.do_predict
        else None
    )

    # 4) We need to store word_ids for eval/test so we can realign predictions later
    # We'll store them in a list of lists
    def extract_word_ids(dataset: NerDataset) -> List[List[Optional[int]]]:
        word_ids_all = []
        for item in dataset:
            # item["word_ids"] is a list of length max_seq_length
            word_ids_all.append(item["word_ids"])  # keep as-is
        return word_ids_all

    word_ids_for_eval = extract_word_ids(eval_dataset) if eval_dataset else None
    word_ids_for_test = extract_word_ids(test_dataset) if test_dataset else None

    # 5) Define compute_metrics function that references the label_map
    def hf_compute_metrics(p: EvalPrediction) -> Dict[str, float]:
        # Depending on whether we are eval or test, pick the right word_ids
        if trainer.eval_dataset is eval_dataset:
            return compute_metrics(p, label_map, word_ids_for_eval)
        else:
            # Fallback or test scenario
            return compute_metrics(p, label_map, word_ids_for_test)

    # 6) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=hf_compute_metrics,
    )

    # 7) Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # 8) Evaluation
    if training_args.do_eval and eval_dataset is not None:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()
        logger.info("Eval results: %s", results)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                for key, value in results.items():
                    writer.write(f"{key} = {value}\n")

    # 9) Prediction
    if training_args.do_predict and test_dataset is not None:
        logger.info("*** Test ***")
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        logger.info("Test metrics: %s", metrics)

        # Realign to word level
        pred_word_labels, gold_word_labels = align_predictions_word_level(
            predictions, label_ids, word_ids_for_test, label_map
        )

        # Save raw metrics
        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write(f"{key} = {value}\n")

            # Write predictions in a new file
            output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
            with open(output_test_predictions_file, "w", encoding="utf-8") as writer:
                for p_words in pred_word_labels:
                    for label_str in p_words:
                        writer.write(f"{label_str}\n")
                    writer.write("\n")  # separate sentences

            logger.info(f"Test predictions saved to {output_test_predictions_file}")

    logger.info("All done!")


if __name__ == "__main__":
    main()
