import random
from typing import List, Literal, Optional


import tiktoken
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.dataset import Dataset

from .padder_msg import PADDINGS_PER_TYPE
from .random_pad_eng_text import padding_texts


def augment_with_random_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    random_text = generate_random_tokens_and_decode(num_tokens=padding_length, seed=seed)
    sample.input =  f"{random_text}\n\n{sample.input}"
    return sample

# -- Adding irrelevant padding in English
def augment_with_non_relevant_padding(sample, padding_length: int, seed: int = 0):
    if padding_length == 0:
        padding_text = ""
    if padding_length == 100:
        padding_text = padding_texts["100"]
    if padding_length == 500:
        padding_text = padding_texts["500"]
    if padding_length == 1000:
        padding_text = padding_texts["1000"]
    if padding_length == 2000:
        padding_text = padding_texts["2000"]
    if padding_length == 5000:
        padding_text = padding_texts["5000"]
    if padding_length == 10000:
        padding_text = padding_texts["10_000"]
    if padding_length == 20000:
        padding_text = padding_texts["20_000"]
    if padding_length == 30000:
        padding_text = padding_texts["30_000"]
    if padding_length == 50000:
        padding_text = padding_texts["50_000"]
    if padding_length == 100000:
        padding_text = padding_texts["100_000"]
    if padding_length == 200000:
        padding_text = padding_texts["200_000"]
    
    # padding before instruction
    # sample.input = f"{padding_text}\n\n{sample.input}" 
    # padding after instruction
    # sample.input = f"{sample.input}\n\n{padding_text}" 
    # padding around instruction
    sample.input = f"{padding_text}\n{sample.input}\n{padding_text}"
    return sample

def augment_with_non_relevant_padding_around_task(sample, padding_length: int, seed: int = 0):
    if padding_length == 0:
        padding_text = ""
    if padding_length == 100:
        padding_text = padding_texts["100"]
    if padding_length == 1000:
        padding_text = padding_texts["500"]
    if padding_length == 2000:
        padding_text = padding_texts["1000"]
    if padding_length == 5000:
        padding_text = padding_texts["2000"] + '\n' + padding_texts["500"]
    if padding_length == 10000:
        padding_text = padding_texts["5000"]
    if padding_length == 20000:
        padding_text = padding_texts["10_000"]
    if padding_length == 30000:
        padding_text = padding_texts["10_000"] + '\n' + padding_texts["5000"]
    if padding_length == 50000:
        padding_text = padding_texts["20_000"] + '\n' + padding_texts["5000"]
    if padding_length == 100000:
        padding_text = padding_texts["50_000"]
    if padding_length == 200000:
        padding_text = padding_texts["100_000"]

    sample.input = f"{padding_text}\n{sample.input}\n{padding_text}"

    return sample

def augment_with_non_relevant_text_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    paddings_to_sample: list[str] = PADDINGS_PER_TYPE["non_relevant"]
    if seed is not None:
        random.seed(seed)
    sampled_padding = random.choice(paddings_to_sample)
    # TODO: Add logic to handle different padding_length
    # TODO: based on implementation method, you probably need to added warning if padding_length > maximum available padding length
    sample.input =  f"{sampled_padding}\n\n{sample.input}"
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
    "non_relevant": augment_with_non_relevant_padding,
    "task_relevant": augment_with_task_relevant_text_padding,
    "non_relevant_around": augment_with_non_relevant_padding_around_task,
}

# For some reason this tokens couldn't be decoded and raise error encding.decode([BLACKLIST[i]])
BLACKLIST_RANDOM_TOKENS_ID = set(
    [199998,200000,200001,200002,200003,200004, 200005,200006, 200007,
    200008, 200009, 200010, 200011,200012, 200013,200014,200015,200016,200017])

def generate_random_tokens_and_decode(num_tokens: int, model="gpt-4o-mini", seed=0):

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


def augment_with_padding(dataset: Dataset, padding_type: Literal["random"]| None, padding_length: int | None, seed: int = 0) -> Dataset:
    if padding_type is None or padding_length is None or padding_length == 0:
        return dataset
    
    aug_func = PADDING_TYPE_TO_FUNC[padding_type]    
    aug_dataset = []
    for sample in dataset:
        aug_dataset.append(aug_func(sample, padding_length, seed=seed))
    return MemoryDataset(aug_dataset)
        
if __name__ == "__main__":
    # For debugging purposes only
    from dataclasses import dataclass
    from typing import Any
    import re

    @dataclass
    class Sample:
        input: str
        
    sample = Sample(input = "abc")

    padding_lengths = [0, 100, 1000, 2000, 5000, 10000, 20000]
    for padding_length in padding_lengths:
        sample = augment_with_non_relevant_padding(sample, padding_length=padding_length)
        text = sample.input
        # \w+ matches one or more word characters (letters, digits, underscores)
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        words_ = text.split()
        word_count_ = len(words_)

        print(f"Padding Length: {padding_length}\nWord Count (Regex): {word_count}\nWord Count (WC): {word_count_}")