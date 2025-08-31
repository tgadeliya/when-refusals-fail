import json
import random
from pathlib import Path
from typing import Literal

from inspect_ai import task
import pandas as pd
import tiktoken
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from .padder_msg import PADDINGS_PER_TYPE
from .task_relevant_text import CATEGORIES, build_padding


MULTI_TASK_PADDING_POS_TYPES = Literal["before", "after", "around", "before_with_task_instructed", "around_with_task_instructed"]


def augment_with_random_padding(
    sample: Sample,
    padding_length: int,
    padding_position: Literal["before", "after", "around"] = "before",
    random_seed: int = 0,
) -> Sample:
    random_text = generate_random_tokens_and_decode(
        num_tokens=padding_length, seed=random_seed
    )
    if padding_position == "before":
        sample.input = f"{random_text}\n\n{sample.input}"
    elif padding_position == "after":
        sample.input = f"{sample.input}\n\n{random_text}"
    elif padding_position == "around":
        # Split random_text in the middle and insert input between two parts
        middle_idx = len(random_text) // 2
        sample.input = f"{random_text[:middle_idx]}\n\n{sample.input}\n\n{random_text[middle_idx:]}"
    else:
        raise ValueError(f"Unknown padding position: {padding_position}")
    return sample


def augment_with_non_relevant_text_padding(
    sample: Sample, padding_length: int, seed: int = 0
) -> Sample:
    paddings_to_sample: list[str] = PADDINGS_PER_TYPE["non_relevant"]
    if seed is not None:
        random.seed(seed)
    sampled_padding = random.choice(paddings_to_sample)

    # TODO: Add logic to handle different padding_length

    if padding_length == 0:
        padding_text = ""
    if padding_length == 100:
        padding_text = build_padding["100"]
    if padding_length == 1000:
        padding_text = build_padding["1000"]
    if padding_length == 10000:
        padding_text = build_padding["10000"]
    if padding_length == 50000:
        padding_text = build_padding["50000"]
    if padding_length == 100000:
        padding_text = build_padding["100000"]
    if padding_length == 200000:
        padding_text = build_padding["200000"]
    # TODO: based on implementation method, you probably need to added warning if padding_length > maximum available padding length
    sample.input = f"{sampled_padding}\n\n{sample.input}"
    return sample


def augment_with_task_relevant_text_padding(
    sample: Sample, padding_length: int, seed: int = 0
) -> Sample:
    if seed is not None:
        random.seed(seed)
    if sample.metadata:
        category = sample.metadata["category"] # e.g. disinformation, hate, drugs_use
        name = sample.metadata["name"] # concrete behavior, e.g. 'Propaganda AI Video - Benign' 
    
    # TODO: check whether we could have situation when category and name is not availble. If yes(I doubt) handle appropriately        
    paddings_to_sample = PADDINGS_PER_TYPE["task_relevant"][category][name][padding_length]
    # TODO: Add logic to handle different padding_length
   

        # Use the category on the sample to select the correct corpus
    category = sample.metadata["category"]  # e.g., "Disinformation", "Hate", "Drugs", ... check if its a string
    pad_text = CATEGORIES[category]

    # Build ~padding_length tokens of padding from that corpus
    padding_text = build_padding(pad_text, padding_length)

    # Prepend padding to the input
    sample.input = f"{padding_text}\n\n{sample.input}"
    return sample
    # TODO: based on implementation method, you probably need to added warning if padding_length > maximum available padding length
    sampled_padding = random.choice(paddings_to_sample)
    sample.input =  f"{sampled_padding}\n\n{sample.input}"
    return sample


PADDING_TYPE_TO_FUNC = {
    "random": augment_with_random_padding,
    "non_relevant": augment_with_non_relevant_text_padding,
    "task_relevant": augment_with_task_relevant_text_padding,
    "multi_task": augment_with_multi_task_padding,
}

# For some reason this tokens couldn't be decoded and raise error encding.decode([BLACKLIST[i]])
BLACKLIST_RANDOM_TOKENS_ID = set(
    [
        199998,
        200000,
        200001,
        200002,
        200003,
        200004,
        200005,
        200006,
        200007,
        200008,
        200009,
        200010,
        200011,
        200012,
        200013,
        200014,
        200015,
        200016,
        200017,
    ]
)


def generate_random_tokens_and_decode(
    num_tokens: int, model="gpt-4o-mini", seed=0
) -> str:
    if seed is not None:
        random.seed(seed)
    encoding = tiktoken.encoding_for_model(model)
    vocab_size = encoding.n_vocab

    tokens = []
    for _ in range(num_tokens):
        token = random.randint(0, vocab_size - 1)
        while token in BLACKLIST_RANDOM_TOKENS_ID:
            token = random.randint(0, vocab_size - 1)
        tokens.append(token)

    decoded = encoding.decode(tokens)
    return decoded


def augment_with_padding(
    dataset: Dataset,
    padding_type: Literal["random", "multi_task"] | None,
    padding_length: int | None,
    padding_kwargs: dict = {},
    random_seed: int = 0,
) -> Dataset:
    if padding_type is None or padding_length is None or padding_length == 0:
        return dataset

    aug_func = PADDING_TYPE_TO_FUNC[padding_type]
    aug_dataset = []
    for sample in dataset:
        aug_dataset.append(
            aug_func(
                sample,
                padding_length=padding_length,
                random_seed=random_seed,
                **padding_kwargs,
            )
        )
    return MemoryDataset(aug_dataset)
