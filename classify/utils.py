import torch
import random
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import pandas as pd
import logging
from comp9312.classify.data import Segment

logger = logging.getLogger(__name__)


def parse_data_files(data_paths):
    """
    Parse data files (assumed to have 'sentence1', 'sentence2', and 'label' columns)
    and return list of segments for each file and index the labels.
    :param data_paths: tuple(Path) - tuple of filenames
    :return: tuple( [[Segment, ...], [Segment, ...]], label_vocab )
    """
    datasets, labels = list(), list()

    for data_path in data_paths:
        df = pd.read_csv(data_path)  # Assuming CSV, adjust if needed
        dataset = [Segment(sentence1=kwargs['sentence1'], sentence2=kwargs['sentence2'], label=kwargs['label'])
                   for kwargs in df.to_dict(orient="records")]
        datasets.append(dataset)
        labels += [segment.label for segment in dataset]

    # Generate vocabs for tags (labels)
    counter = Counter(labels)
    counter = OrderedDict(sorted(counter.items(), key=lambda x: x[1], reverse=True))
    label_vocab = vocab(counter)
    return tuple(datasets), label_vocab


def set_seed(seed):
    """
    Set the seed for random initialization and set
    CUDANN parameters to ensure deterministic results across
    multiple runs with the same seed

    :param seed: int
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False