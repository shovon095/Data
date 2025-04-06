# utils_ner.py
# ----------------------------------------
# A robust utility for Named Entity Recognition
# using Hugging Face fast tokenizers and careful alignment.
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
# Data classes
# --------------------------------------------------------------------
@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]] = None

@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    word_ids: Optional[List[Optional[int]]] = None
    words: Optional[List[str]] = None  # Store original words

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

# --------------------------------------------------------------------
# Read raw data
# --------------------------------------------------------------------
def read_examples_from_file(data_dir: str, mode: Union[Split, str]) -> List[InputExample]:
    mode = mode.value if isinstance(mode, Split) else mode
    file_path = os.path.join(data_dir, f"{mode}.txt")
    examples = []
    guid_index = 1
    words, labels = [], []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    examples.append(InputExample(f"{mode}-{guid_index}", words, labels))
                    guid_index += 1
                    words, labels = [], []
            else:
                splits = line.split()
                words.append(splits[0])
                labels.append(splits[-1] if len(splits) > 1 else "O")

        if words:
            examples.append(InputExample(f"{mode}-{guid_index}", words, labels))

    return examples

# --------------------------------------------------------------------
# Convert examples to features
# --------------------------------------------------------------------
def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_label_id=-100,
) -> List[InputFeatures]:
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []

    for example in examples:
        encoding = tokenizer(
            example.words,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_seq_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        token_type_ids = encoding["token_type_ids"][0].tolist() if "token_type_ids" in encoding else None
        word_ids = encoding.word_ids(batch_index=0)

        label_ids = []
        prev_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(pad_token_label_id)
            elif word_idx != prev_word_idx:
                label_ids.append(label_map[example.labels[word_idx]])
            else:
                label_ids.append(pad_token_label_id)
            prev_word_idx = word_idx

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_ids=label_ids,
            word_ids=word_ids,
            words=example.words,  # Store original words
        ))

    return features

# --------------------------------------------------------------------
# PyTorch dataset
# --------------------------------------------------------------------
class NerDataset(Dataset):
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, labels: List[str],
                 max_seq_length: int, overwrite_cache: bool = False, mode: Split = Split.train):

        cached_features_file = os.path.join(
            data_dir, f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}"
        )

        with FileLock(cached_features_file + ".lock"):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                self.features = torch.load(cached_features_file)
            else:
                examples = read_examples_from_file(data_dir, mode)
                self.features = convert_examples_to_features(
                    examples, labels, max_seq_length, tokenizer, self.pad_token_label_id
                )
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        item = {
            "input_ids": torch.tensor(feat.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feat.attention_mask, dtype=torch.long),
            "labels": torch.tensor(feat.label_ids, dtype=torch.long),
            "word_ids": feat.word_ids,
            "words": feat.words,  # Return original words for inference
        }
        if feat.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(feat.token_type_ids, dtype=torch.long)
        return item

# --------------------------------------------------------------------
# Label loading
# --------------------------------------------------------------------
def get_labels(path: Optional[str] = None) -> List[str]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = f.read().splitlines()
        return ["O"] + labels if "O" not in labels else labels
    else:
        return ["O", "B-PER", "I-PER", "B-ORG", "I-ORG"]
