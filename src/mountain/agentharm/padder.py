import json
import random
from typing import List, Literal, Optional

from inspect_ai import task
import pandas as pd
import tiktoken
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

from .padder_msg import PADDINGS_PER_TYPE
from .non_relevant_padding_text import padding_texts


def augment_with_random_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    random_text = generate_random_tokens_and_decode(num_tokens=padding_length, seed=seed)
    sample.input =  f"{random_text}\n\n{sample.input}"
    return sample

def augment_with_non_relevant_text_padding(
        sample: Sample,
        padding_length: int,
        random_seed: int = 0,
        padding_position: Literal["before", "after", "around"] = "before",
        ) -> Sample:
    if random_seed is not None:
        random.seed(random_seed)
    try:
        padding_text = padding_texts[str(padding_length)]
    except KeyError:
        print(f"Text for the padding: {padding_length} not found.\nPadding text available only for (0, 100, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 100000, 200000) lengths")
    
    if padding_position == "before":
        sample.input = f"{padding_text}\n\n{sample.input}"
    elif padding_position == "after":
        sample.input = f"{sample.input}\n\n{padding_text}"
    elif padding_position == "around":
        if padding_length == 0:
            left_text, right_text = "", ""
        else:
            split_point = len(padding_text) // 2
            left_text = padding_text[:split_point]
            right_text = padding_text[split_point:]
        
        sample.input = f"{left_text}\n\n{sample.input}\n\n{right_text}"
    
    return sample

def augment_with_task_relevant_text_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    if seed is not None:
        random.seed(seed)
    if sample.metadata:
        category = sample.metadata["category"] # e.g. disinformation, hate, drugs_use
        name = sample.metadata["name"] # concrete behavior, e.g. 'Propaganda AI Video - Benign' 
    
    # TODO: check whether we could have situation when category and name is not availble. If yes(I doubt) handle appropriately        
    paddings_to_sample = PADDINGS_PER_TYPE["task_relevant"][category][name][padding_length]
    # TODO: Add logic to handle different padding_length
    # TODO: based on implementation method, you probably need to added warning if padding_length > maximum available padding length
    sampled_padding = random.choice(paddings_to_sample)
    sample.input =  f"{sampled_padding}\n\n{sample.input}"
    return sample

PADDING_TYPE_TO_FUNC = {
    "random": augment_with_random_padding,
    "non_relevant": augment_with_non_relevant_text_padding,
    "task_relevant": augment_with_task_relevant_text_padding,
    # "multi_task": augment_with_multi_task_padding,
}

# For some reason this tokens couldn't be decoded and raise error encding.decode([BLACKLIST[i]])
BLACKLIST_RANDOM_TOKENS_ID = set(
    [199998,200000,200001,200002,200003,200004, 200005,200006, 200007,
    200008, 200009, 200010, 200011,200012, 200013,200014,200015,200016,200017])


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
        dataset: Dataset, padding_type: Literal["random"]| None, padding_length: int | None,
        padding_kwargs: dict = {},
        random_seed: int = 0,) -> Dataset:
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
        
if __name__ == "__main__":
    # For debugging purpose only
    from dataclasses import dataclass
    from typing import Any
    import re

    @dataclass
    class Sample:
        input: str
        
    sample = Sample(input = "abc")

    padding_lengths = [0, 100, 1000, 2000, 5000, 10000, 20000]
    for padding_length in padding_lengths:
        sample = augment_with_non_relevant_text_padding(sample, padding_length=padding_length)
        text = sample.input
        # \w+ matches one or more word characters (letters, digits, underscores)
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        words_ = text.split()
        word_count_ = len(words_)

        print(f"Padding Length: {padding_length}\nWord Count (Regex): {word_count}\nWord Count (WC): {word_count_}")