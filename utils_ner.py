# utils_ner.py
# ----------------------------------------
# A modernized utility for Named Entity Recognition
# using the Hugging Face fast tokenizers with word_ids().
# ----------------------------------------
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from transformers import PreTrainedTokenizer
from filelock import FileLock

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# (A) Data classes
# --------------------------------------------------------------------
@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid:  Unique id for the example.
        words: The words of the sequence.
        labels: (Optional) The labels for each word of the sequence.
                This is usually provided for train and dev sets.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None


@dataclass
class InputFeatures:
    """
    A single set of features of data (tokenized).
    Property names correspond to the model inputs.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

    # Optional: store the word_ids so we can realign at inference
    word_ids: Optional[List[Optional[int]]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


# --------------------------------------------------------------------
# (B) Reading raw data: CoNLL-style
# --------------------------------------------------------------------
def read_examples_from_file(data_dir: str, mode: Union[Split, str]) -> List[InputExample]:
    """
    Reads a file like 'train.txt' or 'dev.txt' or 'test.txt' in CoNLL format:
      Each line: "token label"
      Blank lines separate sentences
    """
    if isinstance(mode, Split):
        mode = mode.value  # e.g. "train", "dev", "test"

    file_path = os.path.join(data_dir, f"{mode}.txt")
    logger.info(f"Reading {mode} data from {file_path}")

    examples = []
    guid_index = 1
    words: List[str] = []
    labels: List[str] = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                # blank line or special marker -> new sentence
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                # typical line: "token label"
                splits = line.split()
                word = splits[0]
                words.append(word)
                if len(splits) > 1:
                    label = splits[-1]
                else:
                    # If no label is provided (e.g. test set), default to "O"
                    label = "O"
                labels.append(label)

        # last sentence if no trailing blank line
        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))

    return examples


# --------------------------------------------------------------------
# (C) Convert examples -> features using fast tokenizer with word_ids
# --------------------------------------------------------------------
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_label_id=-100,  # default ignore_index
) -> List[InputFeatures]:
    """
    Convert a list of InputExamples into InputFeatures by using
    `is_split_into_words=True` to track word boundaries.
    """
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for (ex_index, example) in enumerate(examples):
        if ex_index < 5:
            logger.debug(f"Example {ex_index} -> {example}")

        # 'words' = list of raw words from one sentence
        words = example.words
        # 'labels' = same length as words
        word_labels = example.labels if example.labels is not None else None

        # We call the fast tokenizer
        encoding = tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",  # returns PyTorch tensors
        )

        # shape [1, seq_len]
        input_ids = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        token_type_ids = encoding["token_type_ids"][0].tolist() if "token_type_ids" in encoding else None

        # word_ids: list of length seq_len
        word_ids = encoding.word_ids(batch_index=0)

        # Align labels
        label_ids = []
        if word_labels is not None:
            # We have labels, so let's align them to subwords
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(pad_token_label_id)  # special token
                else:
                    # use the label of the original word
                    label_ids.append(label_map[word_labels[word_id]])
        else:
            # no labels (e.g., test set)
            label_ids = [pad_token_label_id if w_id is None else 0 for w_id in word_ids]

        # Store
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label_ids=label_ids,
                word_ids=word_ids,
            )
        )

    return features


# --------------------------------------------------------------------
# (D) Final PyTorch dataset
# --------------------------------------------------------------------
class NerDataset(Dataset):
    """
    A dataset for NER that uses a Hugging Face fast tokenizer
    to store word_ids for easy alignment.
    """
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        max_seq_length: int,
        overwrite_cache: bool = False,
        mode: Split = Split.train,
    ):
        # We’ll cache features to speed up repeated runs
        # e.g. "cached_train_BertTokenizer_128" in data_dir
        cached_features_file = os.path.join(
            data_dir,
            f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}"
        )

        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                examples = read_examples_from_file(data_dir, mode)
                self.features = convert_examples_to_features(
                    examples,
                    label_list=labels,
                    max_seq_length=max_seq_length,
                    tokenizer=tokenizer,
                    pad_token_label_id=self.pad_token_label_id,
                )
                logger.info(f"Saving features into cached file {cached_features_file}")
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        # Return everything as torch tensors
        item = {
            "input_ids": torch.tensor(feat.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feat.attention_mask, dtype=torch.long),
            "labels": torch.tensor(feat.label_ids, dtype=torch.long),
        }
        if feat.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(feat.token_type_ids, dtype=torch.long)

        # Optionally we can pass word_ids for post-processing
        # (though HF's Trainer won't automatically handle it)
        item["word_ids"] = feat.word_ids  # keeping it as a list (not tensor)
        return item


def get_labels(path: Optional[str] = None) -> List[str]:
    """
    Reads a file of labels (one per line), or returns a default set if None.
    Adjust as needed for your label set.
    """
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        # A default set if you have none
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
