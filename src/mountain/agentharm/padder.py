import random
from typing import List, Literal, Optional


import tiktoken
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.dataset import Dataset

from .padder_msg import PADDINGS_PER_TYPE



def augment_with_random_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    random_text = generate_random_tokens_and_decode(num_tokens=padding_length, seed=seed)
    sample.input =  f"{random_text}\n\n{sample.input}"
    return sample

def augment_with_non_relevant_text_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    paddings_to_sample = PADDINGS_PER_TYPE["non_relevant"][padding_length]
    if seed is not None:
        random.seed(seed)
    sampled_padding = random.choice(paddings_to_sample)
    sample.input =  f"{sampled_padding}\n\n{sample.input}"
    return sample

def augment_with_task_relevant_text_padding(sample: Sample, padding_length: int, seed: int = 0) -> Sample:
    if seed is not None:
        random.seed(seed)

    # TODO: Add guarantee for padding length
    category = sample.metadata["category"] # e.g. disinformation, hate, drugs_use
    name = sample.metadata["name"] # concrete behavior, e.g. 'Propaganda AI Video - Benign' 

    paddings_to_sample = PADDINGS_PER_TYPE["task_relevant"][category][name][padding_length]
    sampled_padding = random.choice(paddings_to_sample)
    sample.input =  f"{sampled_padding}\n\n{sample.input}"
    return sample

PADDING_TYPE_TO_FUNC = {
    "random": augment_with_random_padding,
    "non_relevant": augment_with_non_relevant_text_padding,
    "task_relevant": augment_with_task_relevant_text_padding
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
    if padding_type is None or padding_length is None:
        return dataset
    
    aug_func = PADDING_TYPE_TO_FUNC[padding_type]    
    aug_dataset = []
    for sample in dataset:
        aug_dataset.append(aug_func(sample, padding_length, seed=seed))
    return MemoryDataset(aug_dataset)


