from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from transformers import BertTokenizer
from functools import partial


class Segment:
    def __init__(self, sentence1=None, sentence2=None, label=None, pred=None):
        """
        Segment object to hold sentence pairs and label.
        :param sentence1: str
        :param sentence2: str
        :param label: str
        :param pred: str
        """
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.pred = pred


class BertSeqTransform:
    def __init__(self, bert_model, vocab, max_seq_len=512):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.encoder = partial(
            self.tokenizer.encode_plus,  # Use encode_plus for explicit handling of pairs
            max_length=max_seq_len,
            truncation=True,
            add_special_tokens=True,
            padding='max_length',  # Add padding here for consistent length
            return_tensors='pt'
        )
        self.max_seq_len = max_seq_len
        self.vocab = vocab

    def __call__(self, segment):
        encoding = self.encoder(segment.sentence1, segment.sentence2)
        subwords = encoding['input_ids'].squeeze()
        mask = encoding['attention_mask'].squeeze()
        label = self.vocab[segment.label]
        return subwords, label, mask


class DefaultDataset(Dataset):
    def __init__(
        self,
        segments=None,
        vocab=None,
        bert_model="aubmindlab/bert-base-arabertv2",
        max_seq_len=512,
    ):
        """
        The dataset that used to transform the segments into training data
        :param segments: list[Segment] - list of Segment objects with sentence1, sentence2, and label
        :param vocab: vocab object containing indexed tags (labels)
        :param bert_model: str - BERT model
        :param max_seq_len: int - maximum sequence length
        """
        self.transform = BertSeqTransform(bert_model, vocab, max_seq_len=max_seq_len)
        self.segments = segments
        self.vocab = vocab

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, item):
        subwords, label, mask = self.transform(self.segments[item])
        return subwords, label, mask, self.segments[item]

    def collate_fn(self, batch):
        """
        Collate function that is called when the batch is called by the trainer
        :param batch: Dataloader batch
        :return: Same output as the __getitem__ function
        """
        subwords, labels, masks, segments = zip(*batch)

        # Pad sequences in this batch
        subwords = pad_sequence(subwords, batch_first=True, padding_value=0)
        masks = pad_sequence(masks, batch_first=True, padding_value=0)
        return subwords, torch.LongTensor(labels), torch.FloatTensor(masks), segments