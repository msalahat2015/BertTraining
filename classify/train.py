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
    parser = argparse.ArgumentParser(description="Training script for your model")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=5, help="Maximum number of training epochs")
    parser.add_argument("--bert_model", type=str, default="aubmindlab/bert-base-arabertv2", help="Pre-trained BERT model name")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for optimizer")
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
    print("Set the seed for randomization")
    set_seed(args.seed)

    # Get the datasets and vocab for labels
    datasets, vocab = parse_data_files((args.train_path, args.val_path, args.test_path))

    # From the datasets generate the dataloaders
    print("From the datasets generate the dataloaders")
    datasets = [
        DefaultDataset(
            segments=dataset, vocab=vocab, bert_model=args.bert_model
        )
        for dataset in datasets
    ]
    print(datasets)
    print("start gen dataloader")
    shuffle = (True, False, False)
    train_dataloader, val_dataloader, test_dataloader = [DataLoader(
        dataset=dataset,
        shuffle=shuffle[i],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
    ) for i, dataset in enumerate(datasets)]

    # Initialize the model
    print("# Initialize the model")
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
    print("# Initialize the optimizer")
    optimizer = torch.optim.AdamW(lr=args.learning_rate, params=model.parameters())

    # Initialize the loss function
    print("# Initialize the loss function")
    loss = torch.nn.CrossEntropyLoss()
    
    print("# Initialize the trainer")
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
    print("# Train..")
    trainer.train()
    print("# End Train..")
    return


if __name__ == "__main__":
    args = parse_args()
    print(f"Output Path: {args.output_path}")
    print(f"Train Path: {args.train_path}")
    print(f"GPUs: {args.gpus}")
    print(f"Batch Size: {args.batch_size}")
    main(parse_args())