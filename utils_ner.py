# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

logger = logging.getLogger(__name__)

@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: Optional[List[str]]

@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None

class Split(Enum):
    train = "train"
    dev = "devel"
    test = "test"

if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class NerDataset(Dataset):
        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            cached_features_file = os.path.join(
                data_dir, f"cached_{mode.value}_{tokenizer.__class__.__name__}_{max_seq_length}"
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
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

if is_tf_available():
    import tensorflow as tf

    class TFNerDataset:
        features: List[InputFeatures]
        pad_token_label_id: int = -1

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            examples = read_examples_from_file(data_dir, mode)
            self.features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )

            def gen():
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask}, ex.label_ids
                    else:
                        yield {
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        }, ex.label_ids

            output_signature = (
                {key: tf.TensorSpec(shape=[None], dtype=tf.int32) for key in tokenizer.model_input_names},
                tf.TensorSpec(shape=[None], dtype=tf.int64)
            )
            self.dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

        def get_dataset(self):
            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
    if isinstance(mode, Split):
        mode = mode.value
    file_path = os.path.join(data_dir, f"{mode}.tsv")
    guid_index = 1
    examples = []
    words = []
    labels = []

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if line_strip == ".":  # updated to match the full stop separator logic
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
                continue
            splits = line_strip.split("\t")
            if len(splits) == 1:
                words.append(splits[0])
                labels.append("O")
            else:
                words.append(splits[0])
                labels.append(splits[1])

        if words:
            examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))

    logger.info(f"Loaded {len(examples)} examples from {file_path}")
    return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        if ex_index < 5:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if word_tokens:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        special_tokens_count = tokenizer.num_special_tokens_to_add()
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join(tokens))
            logger.info("input_ids: %s", " ".join(map(str, input_ids)))
            logger.info("input_mask: %s", " ".join(map(str, input_mask)))
            logger.info("segment_ids: %s", " ".join(map(str, segment_ids)))
            logger.info("label_ids: %s", " ".join(map(str, label_ids)))

        if "token_type_ids" not in tokenizer.model_input_names:
            segment_ids = None

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            label_ids=label_ids,
        ))

    return features

def get_labels(path: str) -> List[str]:
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    else:
        return ["O", "B-Entity", "I-Entity"]
