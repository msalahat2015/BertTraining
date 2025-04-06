import torch
import random
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
import pandas as pd
import logging
from comp9312.classify.data import Segment

logger = logging.getLogger(__name__)


def parse_data_files(file_paths):
    all_data = []
    label_counter = Counter()

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for index, row in df.iterrows():
            sentence1 = row['sentence1']
            sentence2 = row['sentence2']
            label = str(row['label'])  # Convert label to string here
            all_data.append({'sentence1': sentence1, 'sentence2': sentence2, 'label': label})
            label_counter[label] += 1

    label_vocab = vocab(label_counter)
    label_vocab.set_default_index(label_vocab["0"] if "0" in label_vocab else 0) # Set a default index

    return all_data, label_vocab

def set_seed(seed):
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)