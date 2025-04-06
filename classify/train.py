import torch
import os
import argparse
import logging
import sys
from torch.utils.data import DataLoader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comp9312.classify.model import BertClassifier
from comp9312.classify.trainer import BertTrainer
from comp9312.classify.utils import parse_data_files, set_seed
from comp9312.classify.data import DefaultDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_path", type=str, required=True, help="Output path",
    )

    parser.add_argument(
        "--train_path", type=str, required=True, help="Path to training data (must have sentence1, sentence2, label columns)",
    )

    parser.add_argument(
        "--val_path", type=str, required=True, help="Path to validation data (must have sentence1, sentence2, label columns)",
    )

    parser.add_argument(
        "--test_path", type=str, required=True, help="Path to test data (must have sentence1, sentence2, label columns)",
    )

    # ... (other arguments remain the same)

    args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(levelname)s\t%(name)s\t%(asctime)s\t%(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        force=True
    )

    # Set the seed for randomization
    set_seed(args.seed)

    # Get the datasets and vocab for labels
    datasets, vocab = parse_data_files((args.train_path, args.val_path, args.test_path))

    # From the datasets generate the dataloaders
    datasets = [
        DefaultDataset(
            segments=dataset, vocab=vocab, bert_model=args.bert_model
        )
        for dataset in datasets
    ]

    shuffle = (True, False, False)
    train_dataloader, val_dataloader, test_dataloader = [DataLoader(
        dataset=dataset,
        shuffle=shuffle[i],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    ) for i, dataset in enumerate(datasets)]

    # Initialize the model
    model = BertClassifier(
        bert_model=args.bert_model, num_labels=len(vocab), dropout=0.1
    )

    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(gpu) for gpu in range(len(args.gpus))]
        )
        model = torch.nn.DataParallel(model, device_ids=range(len(args.gpus)))
        model = model.cuda()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())

    # Initialize the loss function
    loss = torch.nn.CrossEntropyLoss()

    # Initialize the trainer
    trainer = BertTrainer(
        model=model,
        optimizer=optimizer,
        loss=loss,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        output_path=args.output_path,
        max_epochs=args.max_epochs,
    )
    trainer.train()
    return


if __name__ == "__main__":
    main(parse_args())